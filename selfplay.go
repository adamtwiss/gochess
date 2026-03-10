package chess

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// SelfPlayConfig controls self-play game generation for tuning data.
// TimePerMove and FixedDepth are mutually exclusive: set one, leave the other zero.
type SelfPlayConfig struct {
	TimePerMove  time.Duration // time-limited: search time per move (e.g. 200ms-1s)
	FixedDepth   int           // depth-limited: search to exactly this depth per move
	NumGames     int           // total games to generate
	Threads      int           // search threads per game (Lazy SMP)
	Concurrency  int           // parallel games
	OpeningsFile string        // EPD file with starting positions
	OutputFile   string        // output file for training data (.bin)
	HashMB       int           // TT size in MB per game
	NNUENet      *NNUENet      // optional NNUE network (shared read-only across games)
	SyzygyPath   string        // path to Syzygy tablebases (empty = disabled)
	Book         *OpeningBook  // optional opening book for game diversification
}

// SelfPlayGame holds the result of one self-play game.
type SelfPlayGame struct {
	Records   [][BinpackRecordSize]byte // packed binary training records
	Result    float64                   // game result from White's perspective: 1.0, 0.5, 0.0
	ResultStr string                    // "1-0", "0-1", or "1/2-1/2"
	Plies     int                       // total plies played
}

// NumPositions returns the number of recorded positions.
func (g *SelfPlayGame) NumPositions() int {
	return len(g.Records)
}

// GameOverReason describes why a game ended.
type GameOverReason int

const (
	GameNotOver GameOverReason = iota
	GameCheckmate
	GameStalemate
	GameFiftyMove
	GameThreefold
	GameInsufficient
	GameAdjudication
)

// gameOverResult checks if the game is over, returning the reason and the
// result from White's perspective (1.0, 0.5, 0.0).
func gameOverResult(b *Board, hashCounts map[uint64]int, adjEvalCount int, adjEval int) (GameOverReason, float64) {
	moves := b.GenerateLegalMoves()
	if len(moves) == 0 {
		if b.InCheck() {
			// Checkmate: the side to move lost
			if b.SideToMove == White {
				return GameCheckmate, 0.0
			}
			return GameCheckmate, 1.0
		}
		return GameStalemate, 0.5
	}

	// 50-move rule
	if b.HalfmoveClock >= 100 {
		return GameFiftyMove, 0.5
	}

	// Threefold repetition
	if hashCounts[b.HashKey] >= 3 {
		return GameThreefold, 0.5
	}

	// Insufficient material
	if isInsufficientMaterial(b) {
		return GameInsufficient, 0.5
	}

	// Adjudication: resign if eval exceeds ±1000cp for 5 consecutive moves
	if adjEvalCount >= 5 {
		if adjEval > 1000 {
			return GameAdjudication, 1.0
		}
		if adjEval < -1000 {
			return GameAdjudication, 0.0
		}
	}

	return GameNotOver, 0
}

// isInsufficientMaterial returns true for K vs K, KN vs K, KB vs K.
func isInsufficientMaterial(b *Board) bool {
	// Any pawns, rooks, or queens means sufficient material
	if b.Pieces[WhitePawn] != 0 || b.Pieces[BlackPawn] != 0 {
		return false
	}
	if b.Pieces[WhiteRook] != 0 || b.Pieces[BlackRook] != 0 {
		return false
	}
	if b.Pieces[WhiteQueen] != 0 || b.Pieces[BlackQueen] != 0 {
		return false
	}

	wMinors := b.Pieces[WhiteKnight].Count() + b.Pieces[WhiteBishop].Count()
	bMinors := b.Pieces[BlackKnight].Count() + b.Pieces[BlackBishop].Count()

	// K vs K
	if wMinors == 0 && bMinors == 0 {
		return true
	}
	// KN vs K or KB vs K
	if (wMinors <= 1 && bMinors == 0) || (bMinors <= 1 && wMinors == 0) {
		return true
	}
	return false
}

// PlaySelfPlayGame plays one complete self-play game and returns the recorded positions.
func PlaySelfPlayGame(cfg SelfPlayConfig, startFEN string, rng *rand.Rand) SelfPlayGame {
	var b Board
	if err := b.SetFEN(startFEN); err != nil {
		// Fallback to starting position
		b.Reset()
	}

	// Set up NNUE if configured
	if cfg.NNUENet != nil {
		UseNNUE = true
		b.NNUENet = cfg.NNUENet
		b.NNUEAcc = NewNNUEAccumulatorStack(256)
		cfg.NNUENet.RecomputeAccumulator(b.NNUEAcc.Current(), &b)
	}

	tt := NewTranspositionTable(cfg.HashMB)

	// Determine search mode
	depthLimited := cfg.FixedDepth > 0 && cfg.TimePerMove == 0
	maxDepth := 64
	if depthLimited {
		maxDepth = cfg.FixedDepth
	}

	hashCounts := make(map[uint64]int)
	hashCounts[b.HashKey]++

	// Play book moves for opening diversification.
	// Each game follows a different weighted-random path through the book,
	// creating natural variety without artificial random moves.
	if cfg.Book != nil {
		for {
			bookMove, ok := cfg.Book.PickMoveRng(&b, rng)
			if !ok {
				break // out of book
			}
			b.MakeMove(bookMove)
			hashCounts[b.HashKey]++
		}
	}

	var records [][BinpackRecordSize]byte
	adjEvalCount := 0
	lastEval := 0
	totalPlies := 0

	// Count initial ply offset from the FEN (openings may start after 3 moves = 6 plies)
	// Book moves are included in the offset so early searched positions are still filtered.
	initialPly := (b.FullmoveNum - 1) * 2
	if b.SideToMove == Black {
		initialPly++
	}

	for totalPlies < 600 { // safety cap
		// Check game termination
		reason, result := gameOverResult(&b, hashCounts, adjEvalCount, lastEval)
		if reason != GameNotOver {
			patchBinpackResults(records, result)
			return SelfPlayGame{
				Records:   records,
				Result:    result,
				ResultStr: resultString(result),
				Plies:     totalPlies,
			}
		}

		// Search for best move
		info := &SearchInfo{
			StartTime: time.Now(),
			TT:        tt,
		}
		if depthLimited {
			// Safety timeout: cap any single move at 60s to prevent
			// pathological positions (check/recapture extensions) from stalling
			safetyTimeout := 60 * time.Second
			info.MaxTime = safetyTimeout
			atomic.StoreInt64(&info.Deadline, time.Now().Add(safetyTimeout).UnixNano())
		} else {
			info.MaxTime = cfg.TimePerMove
			deadline := time.Now().Add(cfg.TimePerMove)
			atomic.StoreInt64(&info.Deadline, deadline.UnixNano())
		}

		bestMove, searchResult := b.SearchParallel(maxDepth, info, cfg.Threads)
		if bestMove == NoMove {
			// No move found (shouldn't happen if game isn't over)
			break
		}

		// Track eval for adjudication (side-to-move relative -> White-relative)
		eval := searchResult.Score
		if b.SideToMove == Black {
			eval = -eval
		}

		// Clamp mate/TB scores to trainable range before filtering and recording.
		// TB wins (28800) and mates (29000-ply) become +1000; losses become -1000.
		// This preserves these positions in training data instead of filtering them out,
		// giving the NNUE a "clearly winning/losing" signal for endgames and forced mates.
		if eval > 20000 {
			eval = 1000
		} else if eval < -20000 {
			eval = -1000
		}

		// Adjudication tracking: consecutive moves with |eval| > 1000
		absEval := eval
		if absEval < 0 {
			absEval = -absEval
		}
		if absEval > 1000 {
			adjEvalCount++
			lastEval = eval
		} else {
			adjEvalCount = 0
			lastEval = 0
		}

		// Record position for training data (with filtering)
		// Result byte is 0 here; patched to actual result when game ends.
		plyFromStart := initialPly + totalPlies
		if shouldRecordPosition(&b, bestMove, eval, plyFromStart) {
			rec := PackPosition(&b, int16(eval), 0)
			records = append(records, rec)
		}

		// Make the move
		b.MakeMove(bestMove)
		totalPlies++

		// Update repetition tracking
		hashCounts[b.HashKey]++
	}

	// Game exceeded ply limit — draw
	patchBinpackResults(records, 0.5)
	return SelfPlayGame{
		Records:   records,
		Result:    0.5,
		ResultStr: "1/2-1/2",
		Plies:     totalPlies,
	}
}

// shouldRecordPosition applies filters for training data quality.
// Only records quiet positions where the eval is meaningful without search.
func shouldRecordPosition(b *Board, bestMove Move, whiteRelativeEval int, ply int) bool {
	// Skip first 8 plies (opening book territory)
	if ply < 8 {
		return false
	}
	// Skip positions where side to move is in check
	if b.InCheck() {
		return false
	}
	// Note: mate/TB scores are clamped to ±1000 before reaching here,
	// so no need to filter on score magnitude.
	// Skip positions where best move is a capture (not quiet)
	if b.Squares[bestMove.To()] != Empty || bestMove.Flags() == FlagEnPassant {
		return false
	}
	// Skip positions where best move is a promotion (not quiet)
	if bestMove.IsPromotion() {
		return false
	}
	// Skip positions where best move gives check (not quiet)
	b.MakeMove(bestMove)
	givesCheck := b.InCheck()
	b.UnmakeMove(bestMove)
	if givesCheck {
		return false
	}
	return true
}

// patchBinpackResults sets the result byte (offset 30) in all binpack records.
func patchBinpackResults(records [][BinpackRecordSize]byte, result float64) {
	r := ResultToUint8(result)
	for i := range records {
		records[i][30] = r
	}
}

func resultString(result float64) string {
	if result == 1.0 {
		return "1-0"
	} else if result == 0.0 {
		return "0-1"
	}
	return "1/2-1/2"
}

// LoadOpeningPositions loads FEN strings from an EPD file for self-play opening diversity.
func LoadOpeningPositions(filename string) ([]string, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var positions []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		// Each line is a FEN (possibly with operations after semicolons)
		// Take everything up to the first semicolon
		if idx := strings.IndexByte(line, ';'); idx >= 0 {
			line = strings.TrimSpace(line[:idx])
		}
		positions = append(positions, line)
	}
	return positions, scanner.Err()
}

// RunSelfPlay generates training data by playing self-play games.
// Output is written in binary .bin format (32 bytes per position).
func RunSelfPlay(cfg SelfPlayConfig, onGameDone func(gameNum int, game SelfPlayGame)) error {
	// Initialize Syzygy tablebases if configured
	if cfg.SyzygyPath != "" {
		if SyzygyInit(cfg.SyzygyPath) {
			fmt.Printf("Syzygy tablebases loaded: up to %d-piece positions\n", SyzygyMaxPieceCount())
		} else {
			if !SyzygyCGOAvailable() {
				fmt.Printf("Warning: binary built without CGO, Syzygy tablebases unavailable (install gcc and rebuild)\n")
			} else {
				fmt.Printf("Warning: failed to load Syzygy tablebases from %s\n", cfg.SyzygyPath)
			}
		}
		defer SyzygyFree()
	}

	openings, err := LoadOpeningPositions(cfg.OpeningsFile)
	if err != nil {
		return fmt.Errorf("loading openings: %w", err)
	}
	if len(openings) == 0 {
		return fmt.Errorf("no opening positions found in %s", cfg.OpeningsFile)
	}

	f, err := openBinOutputFile(cfg.OutputFile)
	if err != nil {
		return fmt.Errorf("output file: %w", err)
	}
	defer f.Close()

	writer := bufio.NewWriter(f)
	defer writer.Flush()

	var mu sync.Mutex

	concurrency := cfg.Concurrency
	if concurrency <= 0 {
		concurrency = 1
	}

	gamesDone := int64(0)
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup

	for i := 0; i < cfg.NumGames; i++ {
		sem <- struct{}{}
		wg.Add(1)

		go func(gameNum int) {
			defer wg.Done()
			defer func() { <-sem }()

			// Pick a random opening
			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(gameNum)))
			opening := openings[rng.Intn(len(openings))]

			game := PlaySelfPlayGame(cfg, opening, rng)

			// Write records and invoke callback under lock
			mu.Lock()
			for _, rec := range game.Records {
				writer.Write(rec[:])
			}
			writer.Flush()
			done := atomic.AddInt64(&gamesDone, 1)
			if onGameDone != nil {
				onGameDone(int(done), game)
			}
			mu.Unlock()
		}(i)
	}

	wg.Wait()
	return nil
}

// openBinOutputFile opens or creates a .bin output file for appending.
func openBinOutputFile(path string) (*os.File, error) {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0644)
	if err != nil {
		return nil, err
	}
	// Verify existing file is valid (size must be multiple of record size)
	stat, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	if stat.Size()%BinpackRecordSize != 0 {
		f.Close()
		return nil, fmt.Errorf("%s: existing file size %d is not a multiple of %d", path, stat.Size(), BinpackRecordSize)
	}
	return f, nil
}
