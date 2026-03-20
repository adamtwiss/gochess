package chess

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"
)

// EPDPosition represents a parsed EPD position
type EPDPosition struct {
	FEN         string   // First 4 fields as FEN (no halfmove/fullmove)
	BestMoves   []string // bm - best move(s) in SAN
	AvoidMoves  []string // am - avoid move(s) in SAN
	ID          string   // id - position identifier
	Comments    []string // c0-c9 - comments
	RawOperands map[string]string
}

// ParseEPD parses a single EPD line
func ParseEPD(line string) (*EPDPosition, error) {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "#") {
		return nil, nil // Empty line or comment
	}

	epd := &EPDPosition{
		RawOperands: make(map[string]string),
	}

	// Split into FEN part and operations
	// FEN has 4 fields: position, side, castling, en passant
	fields := strings.Fields(line)
	if len(fields) < 4 {
		return nil, fmt.Errorf("EPD too short: %s", line)
	}

	// Build FEN from first 4 fields
	epd.FEN = strings.Join(fields[:4], " ")

	// Parse operations (everything after the 4 FEN fields)
	// Operations are semicolon-terminated: "bm Qxf7+; id \"test\";"
	opsStr := strings.Join(fields[4:], " ")
	ops := strings.Split(opsStr, ";")

	for _, op := range ops {
		op = strings.TrimSpace(op)
		if op == "" {
			continue
		}

		// Split into opcode and operand
		parts := strings.SplitN(op, " ", 2)
		opcode := parts[0]
		operand := ""
		if len(parts) > 1 {
			operand = strings.TrimSpace(parts[1])
			// Remove quotes from string operands
			operand = strings.Trim(operand, "\"")
		}

		epd.RawOperands[opcode] = operand

		switch opcode {
		case "bm":
			// Best moves - can be multiple, space-separated
			epd.BestMoves = strings.Fields(operand)
		case "am":
			// Avoid moves
			epd.AvoidMoves = strings.Fields(operand)
		case "id":
			epd.ID = operand
		case "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9":
			epd.Comments = append(epd.Comments, operand)
		}
	}

	return epd, nil
}

// LoadEPDFile loads all positions from an EPD file
func LoadEPDFile(filename string) ([]*EPDPosition, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var positions []*EPDPosition
	scanner := bufio.NewScanner(file)
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		epd, err := ParseEPD(scanner.Text())
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNum, err)
		}
		if epd != nil {
			positions = append(positions, epd)
		}
	}

	return positions, scanner.Err()
}

// EPDTestResult holds the result of testing a single EPD position
type EPDTestResult struct {
	Position    *EPDPosition
	BestMove    Move
	BestMoveSAN string
	SearchInfo  SearchInfo
	Passed      bool
	TimeTaken   time.Duration
	SolveDepth int           // depth where correct move was first stably found (0 = never solved)
	SolveTime  time.Duration // elapsed time at that depth
	SolveNodes uint64        // cumulative nodes at that depth

	WeightedScore int  // STS weighted score (0-100), -1 if not applicable
	HasWeighted   bool // true if position had c8/c9 weighted scoring data

	// Hash table stats (probes, hits)
	TTProbes, TTHits     uint64
	PawnProbes, PawnHits uint64
}

// RunEPDTest runs search on an EPD position and checks if it finds a best move.
// It also tracks solve time: the earliest point where the engine found the correct
// move and never switched away.
func RunEPDTest(epd *EPDPosition, depth int, maxTime time.Duration, tt *TranspositionTable) (*EPDTestResult, error) {
	return RunEPDTestWithInfo(epd, depth, maxTime, tt, nil, 1)
}

// RunEPDTestWithInfo is like RunEPDTest but accepts an optional SearchInfo for
// caller-provided callbacks. If info is nil, a default is created. The caller's
// OnDepth callback (if set) is invoked after solve tracking for each depth.
// numThreads controls Lazy SMP parallelism (1 = single-threaded).
func RunEPDTestWithInfo(epd *EPDPosition, depth int, maxTime time.Duration, tt *TranspositionTable, info *SearchInfo, numThreads int) (*EPDTestResult, error) {
	var b Board
	fullFEN := epd.FEN + " 0 1"
	if err := b.SetFEN(fullFEN); err != nil {
		return nil, fmt.Errorf("invalid FEN: %w", err)
	}

	// Attach global NNUE network if available
	if UseNNUE && GlobalNNUENetV5 != nil {
		b.AttachNNUEV5(GlobalNNUENetV5)
	} else if UseNNUE && GlobalNNUENet != nil {
		b.AttachNNUE(GlobalNNUENet)
	}

	// Parse expected best moves from SAN into Move values
	var expectedMoves []Move
	for _, bm := range epd.BestMoves {
		m, err := b.ParseSAN(bm)
		if err == nil {
			expectedMoves = append(expectedMoves, m)
		}
	}

	// Parse avoid moves
	var avoidMoves []Move
	for _, am := range epd.AvoidMoves {
		m, err := b.ParseSAN(am)
		if err == nil {
			avoidMoves = append(avoidMoves, m)
		}
	}

	// Track best move and elapsed time at each depth for solve time computation
	type depthRecord struct {
		move    Move
		elapsed time.Duration
		nodes   uint64
	}
	var records []depthRecord

	start := time.Now()

	if info == nil {
		info = &SearchInfo{}
	}
	info.StartTime = start
	info.MaxTime = maxTime
	info.TT = tt

	// Wrap caller's OnDepth callback
	callerOnDepth := info.OnDepth
	info.OnDepth = func(d, score int, nodes uint64, pv []Move) {
		pvMove := NoMove
		if len(pv) > 0 {
			pvMove = pv[0]
		}
		records = append(records, depthRecord{move: pvMove, elapsed: time.Since(start), nodes: nodes})
		if callerOnDepth != nil {
			callerOnDepth(d, score, nodes, pv)
		}
	}

	bestMove, searchInfo := b.SearchParallel(depth, info, numThreads)
	elapsed := time.Since(start)

	result := &EPDTestResult{
		Position:    epd,
		BestMove:    bestMove,
		BestMoveSAN: b.ToSAN(bestMove),
		SearchInfo:  searchInfo,
		TimeTaken:   elapsed,
	}

	// Collect hash table stats
	if tt != nil {
		result.TTProbes, result.TTHits, _ = tt.Stats()
	}
	if b.PawnTable != nil {
		result.PawnProbes, result.PawnHits = b.PawnTable.Stats()
	}

	// Check pass/fail
	isCorrect := func(m Move) bool {
		for _, em := range expectedMoves {
			if m == em {
				return true
			}
		}
		return false
	}
	isAvoided := func(m Move) bool {
		for _, am := range avoidMoves {
			if m == am {
				return true
			}
		}
		return false
	}

	if isCorrect(bestMove) && !isAvoided(bestMove) {
		result.Passed = true
	}

	// Check for STS weighted scoring (c8/c9 fields)
	if weights := ParseSTSWeights(epd, &b); weights != nil {
		result.HasWeighted = true
		result.WeightedScore = weights[bestMove] // 0 if not in map
	}

	// Compute solve time: walk backwards from the last depth to find
	// the earliest point where the correct move was found and held.
	if result.Passed && len(records) > 0 {
		solveIdx := -1
		for i := len(records) - 1; i >= 0; i-- {
			if isCorrect(records[i].move) && !isAvoided(records[i].move) {
				solveIdx = i
			} else {
				break
			}
		}
		if solveIdx >= 0 {
			result.SolveDepth = solveIdx + 1 // depths are 1-indexed
			result.SolveTime = records[solveIdx].elapsed
			result.SolveNodes = records[solveIdx].nodes
		}
	}

	return result, nil
}

// FormatKNPS formats a node count and duration as human-readable kNPS with
// comma-separated thousands (e.g., "1,234 kNPS").
func FormatKNPS(nodes uint64, elapsed time.Duration) string {
	if elapsed <= 0 {
		return "- kNPS"
	}
	knps := float64(nodes) / elapsed.Seconds() / 1000
	whole := int(knps + 0.5)
	if whole >= 1000 {
		return fmt.Sprintf("%d,%03d kNPS", whole/1000, whole%1000)
	}
	return fmt.Sprintf("%d kNPS", whole)
}

// EPDLogScore computes a continuous quality score for a solved position.
// For solved positions: log2(budget) - log2(cost), where budget is the
// maximum allowed (time limit or total nodes) and cost is when the correct
// move was stably found. Unsolved positions score 0. Higher is better.
func EPDLogScore(budget, cost float64) float64 {
	if cost <= 0 || budget <= 0 {
		return 0
	}
	if cost < 1 {
		cost = 1
	}
	return math.Log2(budget) - math.Log2(cost)
}

// ParseSTSWeights parses STS-style c8 (scores) and c9 (coordinate moves) operands
// into a map from Move to score (0-100). The Board is needed to resolve coordinate
// notation (e.g., "f4f5") to Move values with correct flags.
// Returns nil if c8/c9 are missing or malformed.
func ParseSTSWeights(epd *EPDPosition, b *Board) map[Move]int {
	c8, ok8 := epd.RawOperands["c8"]
	c9, ok9 := epd.RawOperands["c9"]
	if !ok8 || !ok9 {
		return nil
	}

	scores := strings.Fields(c8)
	coords := strings.Fields(c9)
	if len(scores) != len(coords) || len(scores) == 0 {
		return nil
	}

	// Generate all pseudo-legal moves for matching
	allMoves := b.GenerateAllMoves()

	result := make(map[Move]int, len(scores))
	for i, coord := range coords {
		if len(coord) < 4 {
			continue
		}
		from := ParseSquare(coord[0:2])
		to := ParseSquare(coord[2:4])
		if from == NoSquare || to == NoSquare {
			continue
		}

		score := 0
		fmt.Sscanf(scores[i], "%d", &score)

		// Find matching pseudo-legal move (to get correct flags)
		for _, m := range allMoves {
			if m.From() == from && m.To() == to {
				// For promotions in coordinate notation (5+ chars), match promotion piece
				if len(coord) >= 5 {
					promoChar := coord[4]
					switch promoChar {
					case 'q':
						if m.Flags() != FlagPromoteQ {
							continue
						}
					case 'r':
						if m.Flags() != FlagPromoteR {
							continue
						}
					case 'b':
						if m.Flags() != FlagPromoteB {
							continue
						}
					case 'n':
						if m.Flags() != FlagPromoteN {
							continue
						}
					}
				}
				result[m] = score
				break
			}
		}
	}

	if len(result) == 0 {
		return nil
	}
	return result
}

// ExtractSTSTheme extracts the theme name from an STS position ID.
// e.g., "STS(v1.0) Undermine.001" -> "Undermine"
// Returns "" if the ID doesn't match STS format.
func ExtractSTSTheme(id string) string {
	// Strip "STS(vN.N) " prefix
	idx := strings.Index(id, ") ")
	if idx < 0 || !strings.HasPrefix(id, "STS(") {
		return ""
	}
	rest := id[idx+2:] // e.g., "Undermine.001"

	// Strip ".NNN" suffix
	dotIdx := strings.LastIndex(rest, ".")
	if dotIdx < 0 {
		return ""
	}
	theme := rest[:dotIdx]

	// Normalize "Knight Outposts/Centralization/Repositioning" vs
	// "Knight Outposts/Repositioning/Centralization" by sorting slash-separated parts
	if strings.Contains(theme, "/") {
		parts := strings.Split(theme, "/")
		sort.Strings(parts)
		theme = strings.Join(parts, "/")
	}

	return theme
}

// EPDSuiteResult holds the results of running an entire EPD test suite
type EPDSuiteResult struct {
	Results  []*EPDTestResult
	Passed   int
	Failed   int
	Total    int
	Duration time.Duration
}

// RunEPDSuite runs all positions in an EPD file
func RunEPDSuite(filename string, depth int, maxTime time.Duration) (*EPDSuiteResult, error) {
	positions, err := LoadEPDFile(filename)
	if err != nil {
		return nil, err
	}

	tt := NewTranspositionTable(64) // 64MB TT for testing
	suite := &EPDSuiteResult{
		Total: len(positions),
	}

	start := time.Now()

	for _, pos := range positions {
		result, err := RunEPDTest(pos, depth, maxTime, tt)
		if err != nil {
			return nil, fmt.Errorf("position %s: %w", pos.ID, err)
		}

		suite.Results = append(suite.Results, result)
		if result.Passed {
			suite.Passed++
		} else {
			suite.Failed++
		}
	}

	suite.Duration = time.Since(start)
	return suite, nil
}
