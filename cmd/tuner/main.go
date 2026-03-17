package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"chess"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "selfplay":
		runSelfPlay(os.Args[2:])
	case "tune":
		runTune(os.Args[2:])
	case "nnue-train":
		runNNUETrain(os.Args[2:])
	case "convert-net":
		runConvertNet(os.Args[2:])
	case "compare-nets":
		runCompareNets(os.Args[2:])
	case "check-net":
		runCheckNet(os.Args[2:])
	case "rescore":
		runRescore(os.Args[2:])
	case "shuffle":
		runShuffle(os.Args[2:])
	case "dump-binpack":
		runDumpBinpack(os.Args[2:])
	case "convert-binpack":
		runConvertBinpack(os.Args[2:])
	case "convert-bullet":
		runConvertBullet(os.Args[2:])
	default:
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Fprintf(os.Stderr, `Usage: tuner <command> [options]

Commands:
  selfplay     Generate training data from self-play games
  tune         Optimize evaluation parameters from training data
  nnue-train   Train an NNUE network from training data
  convert-net  Convert NNUE net between versions (e.g. v3 single-output → v4 output buckets)
  check-net    Health check: evaluate test positions and flag scale issues
  rescore      Re-search positions in a .bin file and update scores in-place
  shuffle      Fisher-Yates shuffle a .bin file in-place (32-byte records)
  dump-binpack    Decode and print chains from a Stockfish .binpack file
  convert-binpack Convert Stockfish .binpack to internal .bin format

Run 'tuner <command> -h' for command-specific options.
`)
}

func runSelfPlay(args []string) {
	fs := flag.NewFlagSet("selfplay", flag.ExitOnError)
	games := fs.Int("games", 1000, "number of games to play")
	timeMS := fs.Int("time", 0, "time per move in milliseconds (mutually exclusive with -depth)")
	depth := fs.Int("depth", 0, "fixed search depth per move (mutually exclusive with -time)")
	threads := fs.Int("threads", 1, "search threads per game (Lazy SMP)")
	concurrency := fs.Int("concurrency", 1, "number of games to play concurrently")
	openings := fs.String("openings", "testdata/noob_3moves.epd", "EPD file with opening positions")
	output := fs.String("output", "training.bin", "output file for training data (.bin)")
	hashMB := fs.Int("hash", 16, "TT size in MB per game")
	nnueFile := fs.String("nnue", "", "NNUE network file (default: net.nnue in current directory)")
	classical := fs.Bool("classical", false, "disable NNUE, use classical eval only")
	syzygyPath := fs.String("syzygy", "", "path to Syzygy tablebase files")
	bookFile := fs.String("book", "", "Polyglot opening book for game diversification")
	blunderRate := fs.Float64("blunder-rate", 0, "probability of playing a random move (0.0-1.0, creates material imbalances for training)")

	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: tuner selfplay [options]\n\nOptions:\n")
		fs.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nUse -time OR -depth (mutually exclusive). Default is -time 200.\n")
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  tuner selfplay -games 20000 -time 200 -concurrency 6\n")
		fmt.Fprintf(os.Stderr, "  tuner selfplay -games 20000 -time 200 -concurrency 6 -syzygy /path/to/tb\n")
	}
	fs.Parse(args)

	// Validate mutual exclusivity; default to -time 200 if neither specified
	if *timeMS > 0 && *depth > 0 {
		fmt.Fprintf(os.Stderr, "Error: -time and -depth are mutually exclusive. Use one or the other.\n")
		os.Exit(1)
	}
	if *timeMS == 0 && *depth == 0 {
		*timeMS = 200 // default: time-limited at 200ms
	}

	// Load NNUE network (same behavior as chess binary)
	var nnueNet *chess.NNUENet
	if *classical {
		// Explicitly disabled
	} else if *nnueFile != "" {
		net, err := chess.LoadNNUE(*nnueFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading NNUE: %v\n", err)
			os.Exit(1)
		}
		nnueNet = net
		fmt.Printf("NNUE loaded from %s\n", *nnueFile)
	} else {
		const defaultNet = "net.nnue"
		net, err := chess.LoadNNUE(defaultNet)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: %s not found, falling back to classical eval\n", defaultNet)
		} else {
			nnueNet = net
			fmt.Printf("NNUE loaded from %s\n", defaultNet)
		}
	}

	// Load opening book for game diversification
	var book *chess.OpeningBook
	if *bookFile != "" {
		// Explicit path — must exist
		var err error
		book, err = chess.LoadOpeningBook(*bookFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading book: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Opening book loaded: %d positions\n", book.Size())
	} else {
		// Try default books in order of preference
		for _, defaultBook := range []string{"testdata/Titans.bin", "book.bin"} {
			if b, err := chess.LoadOpeningBook(defaultBook); err == nil {
				book = b
				fmt.Printf("Opening book loaded from %s (%d positions)\n", defaultBook, b.Size())
				break
			}
		}
	}

	cfg := chess.SelfPlayConfig{
		TimePerMove:  time.Duration(*timeMS) * time.Millisecond,
		FixedDepth:   *depth,
		NumGames:     *games,
		Threads:      *threads,
		Concurrency:  *concurrency,
		OpeningsFile: *openings,
		OutputFile:   *output,
		HashMB:       *hashMB,
		NNUENet:      nnueNet,
		SyzygyPath:   *syzygyPath,
		Book:         book,
		BlunderRate:  *blunderRate,
	}

	fmt.Printf("Self-play configuration:\n")
	fmt.Printf("  Games:       %d\n", cfg.NumGames)
	if cfg.FixedDepth > 0 {
		fmt.Printf("  Mode:        depth-limited (depth %d)\n", cfg.FixedDepth)
	} else {
		fmt.Printf("  Mode:        time-limited (%v/move)\n", cfg.TimePerMove)
	}
	fmt.Printf("  Threads:     %d (per game)\n", cfg.Threads)
	fmt.Printf("  Concurrency: %d (parallel games)\n", cfg.Concurrency)
	fmt.Printf("  Openings:    %s\n", cfg.OpeningsFile)
	fmt.Printf("  Output:      %s\n", cfg.OutputFile)
	fmt.Printf("  Hash:        %d MB (per game)\n", cfg.HashMB)
	if cfg.NNUENet != nil {
		fmt.Printf("  NNUE:        %s\n", *nnueFile)
	} else {
		fmt.Printf("  Eval:        classical\n")
	}
	if cfg.SyzygyPath != "" {
		fmt.Printf("  Syzygy:      %s\n", cfg.SyzygyPath)
	}
	if cfg.BlunderRate > 0 {
		fmt.Printf("  Blunder:     %.1f%% random moves (plies 0-80)\n", cfg.BlunderRate*100)
	}
	if stat, err := os.Stat(cfg.OutputFile); err == nil {
		fmt.Printf("  (appending to existing file, %d bytes)\n", stat.Size())
	}
	fmt.Println()

	start := time.Now()
	totalPositions := 0

	err := chess.RunSelfPlay(cfg, func(gameNum int, game chess.SelfPlayGame) {
		totalPositions += game.NumPositions()
		elapsed := time.Since(start)
		gps := float64(gameNum) / elapsed.Seconds()
		fmt.Printf("\rGame %d/%d (%s, %d plies, %d positions) [%.1f games/s, %d total positions]        ",
			gameNum, cfg.NumGames, game.ResultStr, game.Plies, game.NumPositions(),
			gps, totalPositions)
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nError: %v\n", err)
		os.Exit(1)
	}

	elapsed := time.Since(start)
	fmt.Printf("\n\nDone: %d games, %d positions in %v (%.1f games/s)\n",
		cfg.NumGames, totalPositions, elapsed.Round(time.Second),
		float64(cfg.NumGames)/elapsed.Seconds())
	fmt.Printf("Output written to %s\n", cfg.OutputFile)
}

func runTune(args []string) {
	fs := flag.NewFlagSet("tune", flag.ExitOnError)
	dataFile := fs.String("data", "training.bin", "training data file (.bin)")
	epochs := fs.Int("epochs", 500, "number of optimization epochs")
	lr := fs.Float64("lr", 1.0, "learning rate")
	l2 := fs.Float64("l2", 0, "L2 regularization strength toward initial values (0=disabled)")
	lambda := fs.Float64("lambda", 0.0, "result vs score weight: 0=score-only (default), 1=result-only")
	fixedK := fs.Float64("K", 0, "fixed sigmoid scaling constant (0=auto-tune)")

	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: tuner tune [options]\n\nOptions:\n")
		fs.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  tuner tune -data training.bin -epochs 1000 -lr 1.0\n")
	}
	fs.Parse(args)

	tuner := chess.NewTuner()
	frozenCount := 0
	for _, f := range tuner.Frozen {
		if f {
			frozenCount++
		}
	}
	fmt.Printf("Parameter count: %d (%d tunable, %d frozen)\n", tuner.NumParams(), tuner.NumParams()-frozenCount, frozenCount)

	// Derive .tbin filename from data filename
	tbinFile := strings.TrimSuffix(*dataFile, ".bin") + ".tbin"

	// Check if .tbin needs to be (re)built
	needBuild := false
	tbinStat, tbinErr := os.Stat(tbinFile)
	if tbinErr != nil {
		needBuild = true
	} else {
		srcStat, srcErr := os.Stat(*dataFile)
		if srcErr == nil && srcStat.ModTime().After(tbinStat.ModTime()) {
			needBuild = true
			fmt.Printf("Source %s is newer than %s, rebuilding cache...\n", *dataFile, tbinFile)
		}
	}

	if needBuild {
		fmt.Printf("Preprocessing %s → %s...\n", *dataFile, tbinFile)
		start := time.Now()
		err := chess.PreprocessBinToFile(tuner, *dataFile, tbinFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error preprocessing: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Preprocessing done in %v\n", time.Since(start).Round(time.Millisecond))
	}

	// Open trace file (mmap)
	tf, err := chess.OpenTraceFile(tbinFile, tuner.NumParams(), tuner.BuildPairMap())
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening trace file: %v\n", err)
		os.Exit(1)
	}
	defer tf.Close()
	fmt.Printf("Trace file: %d train + %d validation positions\n", tf.NumTrain, tf.NumValidation)

	if tf.NumTrain == 0 {
		fmt.Fprintf(os.Stderr, "No training positions in trace file\n")
		os.Exit(1)
	}

	scoreBlend := 1.0 - *lambda
	var K float64
	if *fixedK > 0 {
		K = *fixedK
		fmt.Printf("\nUsing fixed K = %.2f\n", K)
	} else {
		fmt.Printf("\nTuning K (scaling constant)...\n")
		K = tuner.TuneK(tf)
	}
	initialError := tuner.ComputeTrainError(tf, K, scoreBlend)
	initialValError := tuner.ComputeValidationError(tf, K, scoreBlend)
	fmt.Printf("Optimal K = %.2f, initial train error = %.8f, val error = %.8f\n\n", K, initialError, initialValError)

	// Run optimizer
	cfg := chess.DefaultTuneConfig()
	cfg.Epochs = *epochs
	cfg.LR = *lr
	cfg.Lambda = *l2
	cfg.ScoreBlend = scoreBlend

	// Set up SIGINT handler for graceful early stop
	stopCh := make(chan struct{})
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)
	go func() {
		<-sigCh
		fmt.Fprintf(os.Stderr, "\nInterrupted — finishing current epoch and printing parameters...\n")
		close(stopCh)
		signal.Stop(sigCh)
	}()
	cfg.Stop = stopCh

	fmt.Printf("Running Adam optimizer: epochs=%d, lr=%.2f, lambda=%.2f, l2=%.1e\n", cfg.Epochs, cfg.LR, *lambda, cfg.Lambda)
	fmt.Printf("%-8s  %-14s  %-14s  %s\n", "Epoch", "Train Error", "Val Error", "Time")
	fmt.Printf("%-8s  %-14s  %-14s  %s\n", "-----", "-----------", "---------", "----")

	epochStart := time.Now()
	tuner.Tune(tf, K, cfg, func(epoch int, trainErr, valErr float64) {
		elapsed := time.Since(epochStart)
		epochStart = time.Now()
		if epoch <= 10 || epoch%10 == 0 || epoch == cfg.Epochs {
			fmt.Printf("%-8d  %.8f    %.8f  %s\n", epoch, trainErr, valErr, elapsed.Round(time.Millisecond))
		}
	})

	// Print results
	fmt.Printf("\n=== Tuned Parameters ===\n\n")
	w := bufio.NewWriter(os.Stdout)
	tuner.PrintParams(w)
	w.Flush()

	finalError := tuner.ComputeTrainError(tf, K, scoreBlend)
	finalValError := tuner.ComputeValidationError(tf, K, scoreBlend)
	fmt.Printf("\nTrain:      initial=%.8f  final=%.8f  improvement=%.4f%%\n",
		initialError, finalError, (initialError-finalError)/initialError*100)
	fmt.Printf("Validation: initial=%.8f  final=%.8f  improvement=%.4f%%\n",
		initialValError, finalValError, (initialValError-finalValError)/initialValError*100)
}

func runNNUETrain(args []string) {
	fs := flag.NewFlagSet("nnue-train", flag.ExitOnError)
	dataFiles := fs.String("data", "", "training data file(s), comma-separated (.bin)")
	outputFile := fs.String("output", "net.nnue", "output NNUE network file")
	epochs := fs.Int("epochs", 100, "number of training epochs")
	lr := fs.Float64("lr", 0.01, "learning rate")
	batchSize := fs.Int("batch", 16384, "batch size")
	lambda := fs.Float64("lambda", 0.0, "result vs score weight (0=score only [default], 1=result only)")
	kValue := fs.Float64("K", 400, "sigmoid scaling constant (0=auto-tune from data)")
	seed := fs.Int64("seed", 42, "random seed for weight initialization")
	positions := fs.Int("positions", 0, "limit training positions per epoch (0=use all)")
	scaleWeight := fs.Float64("scale-weight", 0.0, "weight for centipawn scale anchoring loss (0=disabled)")
	crossEntropy := fs.Bool("cross-entropy", false, "use cross-entropy loss instead of MSE on sigmoid (stronger gradients)")
	useLAMB := fs.Bool("lamb", false, "use LAMB optimizer instead of plain Adam")
	freezeHidden := fs.Bool("freeze-hidden", false, "only train output bucket weights (freeze input + hidden layers)")
	resumeFile := fs.String("resume", "", "resume training from existing .nnue network file")
	bufferSize := fs.Int("buffer", 1000000, "shuffle buffer size for .binpack files (positions)")
	filterChecks := fs.Bool("filter-checks", false, "skip positions where side to move is in check (use for unfiltered data)")
	sampleRate := fs.Float64("sample", 1.0, "fraction of positions to train on per epoch (0.1 = 10%, default 1.0 = all)")

	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: tuner nnue-train [options]\n\nOptions:\n")
		fs.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  tuner nnue-train -data training.bin -epochs 100 -lr 0.001\n")
		fmt.Fprintf(os.Stderr, "  tuner nnue-train -data a.bin,b.bin -epochs 100 -lr 0.001\n")
		fmt.Fprintf(os.Stderr, "  tuner nnue-train -data test80-2024-01.binpack -epochs 100 -lr 0.001  # Stockfish format\n")
	}
	fs.Parse(args)

	if *dataFiles == "" {
		*dataFiles = "training.bin"
	}

	// Parse comma-separated file list, expanding globs
	paths := expandDataPaths(*dataFiles)

	// Create trainer
	trainer := chess.NewNNUETrainer(*seed)

	// Resume from existing network if specified (accepts both v3 and v4 nets)
	if *resumeFile != "" {
		net, err := chess.LoadNNUEAnyVersion(*resumeFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading resume network: %v\n", err)
			os.Exit(1)
		}
		trainer.LoadWeights(net)
		fmt.Printf("Resumed weights from %s\n", *resumeFile)
	}

	// Use K=400 by default
	actualK := *kValue
	if actualK <= 0 {
		actualK = 400.0
	}

	cfg := chess.NNUETrainConfig{
		Epochs:       *epochs,
		LR:           *lr,
		BatchSize:    *batchSize,
		Lambda:       *lambda,
		K:            actualK,
		MaxPositions: *positions,
		ScaleWeight:  *scaleWeight,
		CrossEntropy: *crossEntropy,
		UseLAMB:      *useLAMB,
		FreezeHidden: *freezeHidden,
	}

	// Set up SIGINT handler — saves best net immediately and exits
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)
	stopCh := make(chan struct{})
	cfg.Stop = stopCh

	// bestNet is shared between the training callback and the signal handler
	var mu sync.Mutex
	var bestNet *chess.NNUENet
	var bestEpoch int
	var bestValLoss float64 = math.MaxFloat64

	go func() {
		<-sigCh
		signal.Stop(sigCh)
		fmt.Fprintf(os.Stderr, "\nInterrupted — saving best network and exiting...\n")
		close(stopCh)
		mu.Lock()
		net, epoch, loss := bestNet, bestEpoch, bestValLoss
		mu.Unlock()
		if net != nil {
			if err := chess.SaveNNUE(*outputFile, net); err == nil {
				fmt.Fprintf(os.Stderr, "Best network (epoch %d, val=%.8f) saved to %s\n", epoch, loss, *outputFile)
			} else {
				fmt.Fprintf(os.Stderr, "Error saving network: %v\n", err)
			}
		} else {
			// No best yet — save current weights
			infNet := chess.QuantizeNetwork(trainer.Net)
			if err := chess.SaveNNUE(*outputFile, infNet); err == nil {
				fmt.Fprintf(os.Stderr, "Network saved to %s (no best checkpoint yet)\n", *outputFile)
			}
		}
		os.Exit(0)
	}()

	runNNUETrainBinpack(trainer, paths, cfg, *outputFile, actualK, *kValue, &mu, &bestNet, &bestEpoch, &bestValLoss, *bufferSize, *filterChecks, *sampleRate)
}

func runNNUETrainBinpack(trainer *chess.NNUETrainer, paths []string, cfg chess.NNUETrainConfig, outputFile string, actualK, requestedK float64, mu *sync.Mutex, bestNet **chess.NNUENet, bestEpoch *int, bestValLoss *float64, bufferSize int, filterChecks bool, sampleRate float64) {
	// Detect format by extension
	var src chess.TrainingDataSource
	if allHaveExtension(paths, ".binpack") {
		sfSrc, err := chess.NewSFBinpackSource(paths, bufferSize, filterChecks)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error opening SF binpack files: %v\n", err)
			os.Exit(1)
		}
		if sampleRate > 0 && sampleRate < 1.0 {
			sfSrc.SampleRate = sampleRate
		}
		src = sfSrc
		fmt.Printf("SF binpack data: ~%d estimated positions from %d file(s) (actual count after epoch 1)\n", src.NumRecords(), len(paths))
		for _, p := range paths {
			stat, _ := os.Stat(p)
			fmt.Printf("  %s (%.1f MB)\n", p, float64(stat.Size())/(1024*1024))
		}
		if bufferSize > 0 {
			fmt.Printf("  Shuffle buffer: %d positions\n", bufferSize)
		}
		if sampleRate > 0 && sampleRate < 1.0 {
			fmt.Printf("  Sampling: %.0f%% of positions per epoch\n", sampleRate*100)
		}
	} else {
		bf, err := chess.OpenBinpackFiles(paths...)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error opening binpack files: %v\n", err)
			os.Exit(1)
		}
		src = bf
		fmt.Printf("Binpack data: %d total positions from %d file(s)\n", src.NumRecords(), len(paths))
		for _, p := range paths {
			stat, _ := os.Stat(p)
			fmt.Printf("  %s (%.1f MB, %d positions)\n", p, float64(stat.Size())/(1024*1024), stat.Size()/chess.BinpackRecordSize)
		}
	}
	defer src.Close()

	if requestedK <= 0 {
		fmt.Printf("\nTuning K (scaling constant)...\n")
		actualK = trainer.TuneKBinpack(src, cfg.Lambda)
		cfg.K = actualK
		fmt.Printf("Using K = %.2f\n", actualK)
	} else {
		fmt.Printf("\nUsing K = %.2f (sigmoid scaling)\n", actualK)
	}

	lossType := "MSE"
	if cfg.CrossEntropy {
		lossType = "cross-entropy"
	}
	if cfg.Lambda == 0 && !cfg.CrossEntropy {
		lossType = "MSE-cp"
	}
	sampleStr := ""
	if sampleRate > 0 && sampleRate < 1.0 {
		sampleStr = fmt.Sprintf(" sample=%.0f%%", sampleRate*100)
	}
	fmt.Printf("\nTraining NNUE: epochs=%d lr=%.4f batch=%d lambda=%.2f loss=%s%s\n",
		cfg.Epochs, cfg.LR, cfg.BatchSize, cfg.Lambda, lossType, sampleStr)
	fmt.Printf("%-8s  %-14s  %-14s  %s\n", "Epoch", "Train Loss", "Val Loss", "Time")
	fmt.Printf("%-8s  %-14s  %-14s  %s\n", "-----", "----------", "--------", "----")

	start := time.Now()
	epochStart := time.Now()
	trainer.TrainBinpack(src, cfg, func(epoch int, trainLoss, valLoss float64, numPositions int) {
		elapsed := time.Since(epochStart)
		epochStart = time.Now()
		marker := ""
		mu.Lock()
		if valLoss < *bestValLoss {
			*bestValLoss = valLoss
			*bestEpoch = epoch
			// Keep best checkpoint in RAM for immediate save on Ctrl+C
			*bestNet = chess.QuantizeNetwork(trainer.Net)
			marker = " *best"
		}
		mu.Unlock()
		if epoch == 1 {
			fmt.Printf("info string %d positions in epoch (%.1fM)\n", numPositions, float64(numPositions)/1e6)
		}
		if epoch <= 10 || epoch%10 == 0 || epoch == cfg.Epochs {
			fmt.Printf("%-8d  %-14.8f  %-14.8f  %s%s\n", epoch, trainLoss, valLoss, elapsed.Round(time.Millisecond), marker)
		}
	})

	elapsed := time.Since(start)
	fmt.Printf("\nTraining completed in %v\n", elapsed.Round(time.Second))

	// Save final weights
	infNet := chess.QuantizeNetwork(trainer.Net)
	if err := chess.SaveNNUE(outputFile, infNet); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving network: %v\n", err)
		os.Exit(1)
	}
	fi, _ := os.Stat(outputFile)
	fmt.Printf("Final network saved to %s (%.1f MB)\n", outputFile, float64(fi.Size())/(1024*1024))

	// Also save best checkpoint to .best file
	mu.Lock()
	bn, be := *bestNet, *bestEpoch
	mu.Unlock()
	if bn != nil {
		bestFile := outputFile + ".best"
		if err := chess.SaveNNUE(bestFile, bn); err == nil {
			fmt.Printf("Best val loss at epoch %d (saved to %s)\n", be, bestFile)
		}
	}
}


func runRescore(args []string) {
	fs := flag.NewFlagSet("rescore", flag.ExitOnError)
	dataFile := fs.String("data", "training.bin", "training data file (.bin) to rescore in-place")
	depth := fs.Int("depth", 8, "fixed search depth for rescoring")
	concurrency := fs.Int("concurrency", 1, "number of parallel workers")
	hashMB := fs.Int("hash", 256, "shared TT size in MB")
	nnueFile := fs.String("nnue", "", "NNUE network file (default: net.nnue)")
	classical := fs.Bool("classical", false, "disable NNUE, use classical eval only")
	syzygyPath := fs.String("syzygy", "", "path to Syzygy tablebase files")

	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: tuner rescore [options]\n\nRe-searches each position in a .bin file at fixed depth and updates the score in-place.\nCrash-safe: restarting rescores from the beginning (idempotent).\n\nOptions:\n")
		fs.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  tuner rescore -data training.bin -depth 10 -concurrency 8 -hash 512\n")
		fmt.Fprintf(os.Stderr, "  tuner rescore -data training.bin -depth 8 -concurrency 4 -syzygy /path/to/tb\n")
	}
	fs.Parse(args)

	// Load NNUE network
	var nnueNet *chess.NNUENet
	if *classical {
		// Explicitly disabled
	} else if *nnueFile != "" {
		net, err := chess.LoadNNUE(*nnueFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading NNUE: %v\n", err)
			os.Exit(1)
		}
		nnueNet = net
		fmt.Printf("NNUE loaded from %s\n", *nnueFile)
	} else {
		const defaultNet = "net.nnue"
		net, err := chess.LoadNNUE(defaultNet)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: %s not found, falling back to classical eval\n", defaultNet)
		} else {
			nnueNet = net
			fmt.Printf("NNUE loaded from %s\n", defaultNet)
		}
	}

	// Initialize Syzygy tablebases if configured
	if *syzygyPath != "" {
		if chess.SyzygyInit(*syzygyPath) {
			fmt.Printf("Syzygy tablebases loaded: up to %d-piece positions\n", chess.SyzygyMaxPieceCount())
		} else {
			if !chess.SyzygyCGOAvailable() {
				fmt.Printf("Warning: binary built without CGO, Syzygy tablebases unavailable\n")
			} else {
				fmt.Printf("Warning: failed to load Syzygy tablebases from %s\n", *syzygyPath)
			}
		}
		defer chess.SyzygyFree()
	}

	// Validate file
	stat, err := os.Stat(*dataFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	totalRecords := stat.Size() / chess.BinpackRecordSize

	cfg := chess.RescoreConfig{
		DataFile:    *dataFile,
		Depth:       *depth,
		Concurrency: *concurrency,
		HashMB:      *hashMB,
		NNUENet:     nnueNet,
		SyzygyPath:  *syzygyPath,
	}

	fmt.Printf("Rescore configuration:\n")
	fmt.Printf("  File:        %s (%d positions, %.1f MB)\n", cfg.DataFile, totalRecords, float64(stat.Size())/(1024*1024))
	fmt.Printf("  Depth:       %d\n", cfg.Depth)
	fmt.Printf("  Concurrency: %d workers\n", cfg.Concurrency)
	fmt.Printf("  Hash:        %d MB (shared)\n", cfg.HashMB)
	if nnueNet != nil {
		fmt.Printf("  Eval:        NNUE\n")
	} else {
		fmt.Printf("  Eval:        classical\n")
	}
	if *syzygyPath != "" {
		fmt.Printf("  Syzygy:      %s\n", *syzygyPath)
	}
	fmt.Println()

	start := time.Now()
	err = chess.RescoreTrainingData(cfg, func(done, total int) {
		elapsed := time.Since(start)
		posPerSec := float64(done) / elapsed.Seconds()
		remaining := time.Duration(float64(total-done)/posPerSec) * time.Second
		fmt.Printf("\r  %d/%d positions (%.0f pos/s, ETA %v)        ",
			done, total, posPerSec, remaining.Round(time.Second))
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nError: %v\n", err)
		os.Exit(1)
	}

	elapsed := time.Since(start)
	fmt.Printf("\n\nDone: %d positions rescored in %v (%.0f pos/s)\n",
		totalRecords, elapsed.Round(time.Second),
		float64(totalRecords)/elapsed.Seconds())
}

func runConvertNet(args []string) {
	fs := flag.NewFlagSet("convert-net", flag.ExitOnError)
	from := fs.String("from", "", "source .nnue file (v3 single-output)")
	to := fs.String("to", "", "destination .nnue file (v4 output buckets)")

	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: tuner convert-net [options]\n\nConverts a v3 NNUE net (single output) to v4 format (8 output buckets).\n")
		fmt.Fprintf(os.Stderr, "Hidden layer weights are preserved; the single output is replicated into all buckets.\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		fs.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExample:\n")
		fmt.Fprintf(os.Stderr, "  tuner convert-net -from net.nnue -to net-v4.nnue\n")
		fmt.Fprintf(os.Stderr, "\nThen resume training to specialize the buckets:\n")
		fmt.Fprintf(os.Stderr, "  tuner nnue-train -data training.bin -resume net-v4.nnue -epochs 100 -lr 0.001\n")
	}
	fs.Parse(args)

	if *from == "" || *to == "" {
		fs.Usage()
		os.Exit(1)
	}

	// Load v3 net (auto-replicates single output into all buckets)
	net, err := chess.LoadNNUEAnyVersion(*from)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading network: %v\n", err)
		os.Exit(1)
	}

	// Save as v4
	if err := chess.SaveNNUE(*to, net); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving network: %v\n", err)
		os.Exit(1)
	}

	fi, _ := os.Stat(*to)
	fmt.Printf("Converted %s → %s (v4 with %d output buckets, %.1f KB)\n",
		*from, *to, chess.NNUEOutputBuckets, float64(fi.Size())/1024)
}

func runCompareNets(args []string) {
	if len(args) != 2 {
		fmt.Fprintf(os.Stderr, "Usage: tuner compare-nets <baseline.nnue> <trained.nnue>\n")
		fmt.Fprintf(os.Stderr, "\nCompares output bucket weights between two v4 NNUE nets.\n")
		os.Exit(1)
	}

	net1, err := chess.LoadNNUEAnyVersion(args[0])
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading %s: %v\n", args[0], err)
		os.Exit(1)
	}
	net2, err := chess.LoadNNUEAnyVersion(args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading %s: %v\n", args[1], err)
		os.Exit(1)
	}

	pieceRanges := [8]string{"2-5", "6-9", "10-13", "14-17", "18-21", "22-25", "26-29", "30-32"}

	fmt.Printf("\n=== Output Bucket Weight Comparison ===\n")
	fmt.Printf("%-8s  %-8s  %-10s  %-10s  %-10s  %-10s  %-8s  %-8s\n",
		"Bucket", "Pieces", "|w| base", "|w| train", "Δ mean", "Δ RMS", "Bias Δ", "Flipped")
	fmt.Printf("%-8s  %-8s  %-10s  %-10s  %-10s  %-10s  %-8s  %-8s\n",
		"------", "------", "--------", "---------", "------", "-----", "------", "-------")

	for b := 0; b < chess.NNUEOutputBuckets; b++ {
		var sumAbs1, sumAbs2, sumDiff, sumDiffSq float64
		flipped := 0
		for j := 0; j < chess.NNUEHidden3Size; j++ {
			w1 := float64(net1.OutputWeights[b][j])
			w2 := float64(net2.OutputWeights[b][j])
			sumAbs1 += math.Abs(w1)
			sumAbs2 += math.Abs(w2)
			d := w2 - w1
			sumDiff += d
			sumDiffSq += d * d
			if (w1 > 0 && w2 < 0) || (w1 < 0 && w2 > 0) {
				flipped++
			}
		}
		n := float64(chess.NNUEHidden3Size)
		biasD := int64(net2.OutputBias[b]) - int64(net1.OutputBias[b])
		fmt.Printf("%-8d  %-8s  %-10.1f  %-10.1f  %+-10.1f  %-10.2f  %+-8d  %d/%d\n",
			b, pieceRanges[b], sumAbs1/n, sumAbs2/n,
			sumDiff/n, math.Sqrt(sumDiffSq/n), biasD, flipped, chess.NNUEHidden3Size)
	}

	// Per-bucket weight distribution
	fmt.Printf("\n=== Output Bucket Weight Distributions (trained net) ===\n")
	fmt.Printf("%-8s  %-8s  %-8s  %-8s  %-8s  %-8s\n",
		"Bucket", "Pieces", "Min", "Max", "Mean", "StdDev")
	fmt.Printf("%-8s  %-8s  %-8s  %-8s  %-8s  %-8s\n",
		"------", "------", "---", "---", "----", "------")
	for b := 0; b < chess.NNUEOutputBuckets; b++ {
		minW, maxW := int16(math.MaxInt16), int16(math.MinInt16)
		var sum float64
		for j := 0; j < chess.NNUEHidden3Size; j++ {
			w := net2.OutputWeights[b][j]
			if w < minW {
				minW = w
			}
			if w > maxW {
				maxW = w
			}
			sum += float64(w)
		}
		mean := sum / float64(chess.NNUEHidden3Size)
		var varSum float64
		for j := 0; j < chess.NNUEHidden3Size; j++ {
			d := float64(net2.OutputWeights[b][j]) - mean
			varSum += d * d
		}
		stdDev := math.Sqrt(varSum / float64(chess.NNUEHidden3Size))
		fmt.Printf("%-8d  %-8s  %-8d  %-8d  %+-8.1f  %-8.1f\n",
			b, pieceRanges[b], minW, maxW, mean, stdDev)
	}

	// Bucket divergence: how different are buckets from each other?
	fmt.Printf("\n=== Inter-Bucket Divergence (RMS between bucket pairs, trained net) ===\n")
	fmt.Printf("%-8s", "")
	for b := 0; b < chess.NNUEOutputBuckets; b++ {
		fmt.Printf("  B%-5d", b)
	}
	fmt.Printf("\n")
	for b1 := 0; b1 < chess.NNUEOutputBuckets; b1++ {
		fmt.Printf("B%-7d", b1)
		for b2 := 0; b2 < chess.NNUEOutputBuckets; b2++ {
			if b2 <= b1 {
				fmt.Printf("  %-6s", "-")
				continue
			}
			var diffSq float64
			for j := 0; j < chess.NNUEHidden3Size; j++ {
				d := float64(net2.OutputWeights[b1][j]) - float64(net2.OutputWeights[b2][j])
				diffSq += d * d
			}
			fmt.Printf("  %-6.1f", math.Sqrt(diffSq/float64(chess.NNUEHidden3Size)))
		}
		fmt.Printf("\n")
	}

	// Also show hidden layer drift summary
	fmt.Printf("\n=== Hidden Layer Drift (RMS of weight differences) ===\n")

	// Hidden2 weights
	var h2DiffSq float64
	for i := range net1.Hidden2Weights {
		for j := range net1.Hidden2Weights[i] {
			d := float64(net2.Hidden2Weights[i][j]) - float64(net1.Hidden2Weights[i][j])
			h2DiffSq += d * d
		}
	}
	h2N := float64(chess.NNUEHidden2Size * chess.NNUEHidden3Size)
	fmt.Printf("Hidden2 weights:  RMS Δ = %.2f\n", math.Sqrt(h2DiffSq/h2N))

	// Hidden1 weights
	var h1DiffSq float64
	for i := range net1.HiddenWeights {
		for j := range net1.HiddenWeights[i] {
			d := float64(net2.HiddenWeights[i][j]) - float64(net1.HiddenWeights[i][j])
			h1DiffSq += d * d
		}
	}
	h1N := float64(chess.NNUEHiddenSize * 2 * chess.NNUEHidden2Size)
	fmt.Printf("Hidden1 weights:  RMS Δ = %.2f\n", math.Sqrt(h1DiffSq/h1N))

	// Input weights (sample first 1000)
	var inDiffSq float64
	inN := 0
	for i := 0; i < chess.NNUEInputSize && i < 1000; i++ {
		for j := range net1.InputWeights[i] {
			d := float64(net2.InputWeights[i][j]) - float64(net1.InputWeights[i][j])
			inDiffSq += d * d
			inN++
		}
	}
	fmt.Printf("Input weights:    RMS Δ = %.2f (sampled first 1000 rows)\n", math.Sqrt(inDiffSq/float64(inN)))
}

func runCheckNet(args []string) {
	fs := flag.NewFlagSet("check-net", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: tuner check-net <net.nnue>\n")
	}
	fs.Parse(args)

	if fs.NArg() == 0 {
		fs.Usage()
		os.Exit(1)
	}

	netPath := fs.Arg(0)

	// Detect version and load appropriate net type
	version, err := chess.DetectNNUEVersion(netPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	var netV4 *chess.NNUENet
	var netV5 *chess.NNUENetV5

	if version == 5 {
		netV5, err = chess.LoadNNUEV5(netPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading v5 net: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "info string NNUE v5 fingerprint %s from %s\n", netV5.Fingerprint(), netPath)
	} else {
		netV4, err = chess.LoadNNUEAnyVersion(netPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	}

	type testPos struct {
		name    string
		fen     string
		expectMin int
		expectMax int
	}

	positions := []testPos{
		{"startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", -50, 50},
		{"miss pawn", "rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1", -200, -20},
		{"miss knight", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1", -500, -80},
		{"miss bishop", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1", -500, -80},
		{"miss rook", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1", -800, -120},
		{"miss queen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1", -1500, -200},
		{"EG rook up", "4k3/8/8/8/8/8/4PPPP/4K2R w K - 0 1", 400, 2500},
		{"EG queen up", "4k3/8/8/8/8/8/4PPPP/3QK3 w - - 0 1", 500, 3000},
	}

	evalPos := func(fen string) int {
		var b chess.Board
		b.Reset()
		b.SetFEN(fen)

		if netV5 != nil {
			acc := chess.NNUEAccumulatorV5{}
			netV5.RecomputeAccumulator(&b, &acc, chess.White)
			netV5.RecomputeAccumulator(&b, &acc, chess.Black)
			return netV5.Forward(&acc, b.SideToMove, b.AllPieces.Count())
		}

		accStack := chess.NewNNUEAccumulatorStack(8)
		b.NNUEAcc = accStack
		b.NNUENet = netV4
		netV4.RecomputeAccumulator(accStack.Current(), &b)
		return netV4.Evaluate(accStack.Current(), b.SideToMove, b.AllPieces.Count())
	}

	fmt.Printf("NNUE Health Check: %s\n\n", netPath)
	fmt.Printf("%-14s  %8s  %8s  %8s  %s\n", "Position", "Score", "Min", "Max", "Status")
	fmt.Printf("%-14s  %8s  %8s  %8s  %s\n", "--------", "--------", "--------", "--------", "------")

	issues := 0
	for _, pos := range positions {
		score := evalPos(pos.fen)
		status := "OK"
		if score < pos.expectMin {
			status = "LOW"
			issues++
		} else if score > pos.expectMax {
			status = "HIGH"
			issues++
		}
		fmt.Printf("%-14s  %8d  %8d  %8d  %s\n", pos.name, score, pos.expectMin, pos.expectMax, status)
	}

	fmt.Println()
	if issues == 0 {
		fmt.Println("All checks passed.")
	} else {
		fmt.Printf("%d issue(s) found — eval scale may be collapsed or miscalibrated.\n", issues)
	}
}

func runShuffle(args []string) {
	fs := flag.NewFlagSet("shuffle", flag.ExitOnError)
	dataFile := fs.String("data", "training.bin", "training data file (.bin) to shuffle in-place")
	seed := fs.Int64("seed", 0, "random seed (0=use current time)")

	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: tuner shuffle [options]\n\nShuffle a .bin training file in-place using Fisher-Yates on 32-byte records.\n\nOptions:\n")
		fs.PrintDefaults()
	}
	fs.Parse(args)

	stat, err := os.Stat(*dataFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	const recSize = chess.BinpackRecordSize // 32 bytes
	if stat.Size()%recSize != 0 {
		fmt.Fprintf(os.Stderr, "Error: file size %d is not a multiple of %d bytes\n", stat.Size(), recSize)
		os.Exit(1)
	}
	n := stat.Size() / recSize

	f, err := os.OpenFile(*dataFile, os.O_RDWR, 0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening file: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	rngSeed := *seed
	if rngSeed == 0 {
		rngSeed = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(rngSeed))

	fmt.Printf("Shuffling %s: %d records (%.1f MB), seed=%d\n", *dataFile, n, float64(stat.Size())/(1024*1024), rngSeed)

	// Fisher-Yates shuffle using ReadAt/WriteAt on 32-byte records
	var buf1, buf2 [recSize]byte
	reportInterval := n / 20
	if reportInterval < 1 {
		reportInterval = 1
	}
	start := time.Now()

	for i := n - 1; i > 0; i-- {
		j := rng.Int63n(i + 1)
		if i == j {
			continue
		}
		// Swap records i and j
		if _, err := f.ReadAt(buf1[:], i*recSize); err != nil {
			fmt.Fprintf(os.Stderr, "\nError reading record %d: %v\n", i, err)
			os.Exit(1)
		}
		if _, err := f.ReadAt(buf2[:], j*recSize); err != nil {
			fmt.Fprintf(os.Stderr, "\nError reading record %d: %v\n", j, err)
			os.Exit(1)
		}
		if _, err := f.WriteAt(buf2[:], i*recSize); err != nil {
			fmt.Fprintf(os.Stderr, "\nError writing record %d: %v\n", i, err)
			os.Exit(1)
		}
		if _, err := f.WriteAt(buf1[:], j*recSize); err != nil {
			fmt.Fprintf(os.Stderr, "\nError writing record %d: %v\n", j, err)
			os.Exit(1)
		}

		if (n-i)%reportInterval == 0 {
			pct := float64(n-i) / float64(n) * 100
			fmt.Printf("\r  %.0f%% complete...", pct)
		}
	}

	elapsed := time.Since(start)
	fmt.Printf("\r  Done. Shuffled %d records in %v\n", n, elapsed.Round(time.Millisecond))
}

func runDumpBinpack(args []string) {
	fs := flag.NewFlagSet("dump-binpack", flag.ExitOnError)
	dataFile := fs.String("data", "", "Stockfish .binpack file to decode")
	numChains := fs.Int("n", 5, "number of chains to dump")

	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: tuner dump-binpack [options]\n\nDecode and print chains from a Stockfish .binpack file for diagnostics.\n\nOptions:\n")
		fs.PrintDefaults()
	}
	fs.Parse(args)

	if *dataFile == "" {
		fs.Usage()
		os.Exit(1)
	}

	if err := chess.DumpSFBinpack(*dataFile, *numChains); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// allHaveExtension checks if all paths share the given extension.
// expandDataPaths splits a comma-separated list of file paths, expanding any
// glob patterns. Quote the argument to prevent shell expansion:
//
//	./tuner nnue-train -data "/training/sf/*.binpack"
func expandDataPaths(dataFiles string) []string {
	parts := strings.Split(dataFiles, ",")
	var paths []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		// Check if path contains glob characters
		if strings.ContainsAny(p, "*?[") {
			matches, err := filepath.Glob(p)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Warning: invalid glob pattern %q: %v\n", p, err)
				continue
			}
			if len(matches) == 0 {
				fmt.Fprintf(os.Stderr, "Warning: glob %q matched no files\n", p)
				continue
			}
			sort.Strings(matches)
			paths = append(paths, matches...)
		} else {
			paths = append(paths, p)
		}
	}
	if len(paths) == 0 {
		fmt.Fprintf(os.Stderr, "Error: no data files found\n")
		os.Exit(1)
	}
	return paths
}

func allHaveExtension(paths []string, ext string) bool {
	for _, p := range paths {
		if !strings.HasSuffix(strings.ToLower(p), ext) {
			return false
		}
	}
	return len(paths) > 0
}

// Ensure path package isn't needed — using strings.HasSuffix above.

func runConvertBinpack(args []string) {
	fs := flag.NewFlagSet("convert-binpack", flag.ExitOnError)
	input := fs.String("input", "", "input Stockfish .binpack file (required)")
	output := fs.String("output", "", "output .bin file (default: input with .bin extension)")
	fs.Parse(args)

	if *input == "" {
		fmt.Fprintln(os.Stderr, "Error: -input is required")
		fs.Usage()
		os.Exit(1)
	}

	outPath := *output
	if outPath == "" {
		outPath = strings.TrimSuffix(*input, ".binpack") + ".bin"
	}

	fmt.Printf("Converting %s -> %s\n", *input, outPath)
	start := time.Now()

	err := chess.ConvertSFBinpack(*input, outPath, func(written, skipped int64) {
		elapsed := time.Since(start)
		rate := float64(written+skipped) / elapsed.Seconds()
		fmt.Printf("\r  %dK written, %dK skipped (%.0f pos/s)   ",
			written/1000, skipped/1000, rate)
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nError: %v\n", err)
		os.Exit(1)
	}

	// Report final stats
	info, _ := os.Stat(outPath)
	fmt.Printf("\nDone in %v. Output: %s (%d positions, %.1f MB)\n",
		time.Since(start).Round(time.Millisecond),
		outPath,
		info.Size()/int64(chess.BinpackRecordSize),
		float64(info.Size())/(1024*1024))
}

func runConvertBullet(args []string) {
	fs := flag.NewFlagSet("convert-bullet", flag.ExitOnError)
	input := fs.String("input", "", "Bullet quantised.bin file (required)")
	output := fs.String("output", "net.nnue", "output .nnue file")
	v5 := fs.Bool("v5", false, "convert as v5 (shallow wide 1024) instead of v4 (deep 256)")
	fs.Parse(args)

	if *input == "" {
		fmt.Fprintln(os.Stderr, "Error: -input is required")
		fs.Usage()
		os.Exit(1)
	}

	data, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading %s: %v\n", *input, err)
		os.Exit(1)
	}

	if *v5 {
		convertBulletV5(data, *output)
		return
	}

	// Expected sizes for our exact architecture (no layerstack buckets on hidden layers)
	const (
		inputSize  = 12288
		hiddenSize = 256
		hidden2    = 32
		hidden3    = 32
		buckets    = 8
	)

	expectedSize := inputSize*hiddenSize*2 + hiddenSize*2 + // l0w + l0b (i16)
		hiddenSize*2*hidden2*2 + hidden2*4 + // l1w (i16) + l1b (i32)
		hidden2*hidden3*2 + hidden3*4 + // l2w (i16) + l2b (i32)
		hidden3*buckets*2 + buckets*4 // l3w (i16) + l3b (i32)

	fmt.Printf("Input file: %d bytes, expected: %d bytes\n", len(data), expectedSize)
	if len(data) < expectedSize {
		fmt.Fprintf(os.Stderr, "Error: file too small (got %d, need %d)\n", len(data), expectedSize)
		os.Exit(1)
	}

	// Parse the Bullet quantised.bin — sequential i16/i32 arrays
	net := &chess.NNUENet{}
	offset := 0

	// l0w: [12288][256] i16
	for i := 0; i < inputSize; i++ {
		for j := 0; j < hiddenSize; j++ {
			net.InputWeights[i][j] = int16(binary.LittleEndian.Uint16(data[offset:]))
			offset += 2
		}
	}

	// l0b: [256] i16
	for j := 0; j < hiddenSize; j++ {
		net.InputBiases[j] = int16(binary.LittleEndian.Uint16(data[offset:]))
		offset += 2
	}

	// l1w: [512][32] i16
	for i := 0; i < hiddenSize*2; i++ {
		for j := 0; j < hidden2; j++ {
			net.HiddenWeights[i][j] = int16(binary.LittleEndian.Uint16(data[offset:]))
			offset += 2
		}
	}

	// l1b: [32] i32
	for j := 0; j < hidden2; j++ {
		net.HiddenBiases[j] = int32(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4
	}

	// l2w: [32][32] i16
	for i := 0; i < hidden2; i++ {
		for j := 0; j < hidden3; j++ {
			net.Hidden2Weights[i][j] = int16(binary.LittleEndian.Uint16(data[offset:]))
			offset += 2
		}
	}

	// l2b: [32] i32
	for j := 0; j < hidden3; j++ {
		net.Hidden2Biases[j] = int32(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4
	}

	// l3w: [32][8] i16 in Bullet -> [8][32] i16 in our format (transpose)
	var l3wRaw [hidden3][buckets]int16
	for i := 0; i < hidden3; i++ {
		for b := 0; b < buckets; b++ {
			l3wRaw[i][b] = int16(binary.LittleEndian.Uint16(data[offset:]))
			offset += 2
		}
	}
	// Transpose to [bucket][hidden3]
	for b := 0; b < buckets; b++ {
		for i := 0; i < hidden3; i++ {
			net.OutputWeights[b][i] = l3wRaw[i][b]
		}
	}

	// l3b: [8] i32
	for b := 0; b < buckets; b++ {
		net.OutputBias[b] = int32(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4
	}

	fmt.Printf("Parsed %d bytes of %d\n", offset, len(data))

	// Prepare transposed weights for SIMD
	net.PrepareWeights()

	// Save in our .nnue format
	if err := chess.SaveNNUE(*output, net); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving: %v\n", err)
		os.Exit(1)
	}

	fi, _ := os.Stat(*output)
	fmt.Printf("Saved %s (%d bytes)\n", *output, fi.Size())
	fmt.Printf("Fingerprint: %s\n", net.Fingerprint())
}

func convertBulletV5(data []byte, outputPath string) {
	const (
		inputSize  = 12288
		hiddenSize = 1024
		buckets    = 8
	)

	// V5 layout from Bullet (with .transpose()):
	// l0w: [inputSize][hiddenSize] i16 (transposed by Bullet SavedFormat)
	// l0b: [hiddenSize] i16
	// l1w: [2*hiddenSize][buckets] i16 (transposed by Bullet SavedFormat)
	// l1b: [buckets] i32
	expectedSize := inputSize*hiddenSize*2 + hiddenSize*2 + // l0w + l0b
		2*hiddenSize*buckets*2 + buckets*4 // l1w + l1b

	fmt.Printf("Input file: %d bytes, expected: %d bytes\n", len(data), expectedSize)
	if len(data) < expectedSize {
		fmt.Fprintf(os.Stderr, "Error: file too small (got %d, need %d)\n", len(data), expectedSize)
		os.Exit(1)
	}

	net := &chess.NNUENetV5{}
	offset := 0

	// l0w: [inputSize][hiddenSize] i16 (already transposed by Bullet)
	for i := 0; i < inputSize; i++ {
		for j := 0; j < hiddenSize; j++ {
			net.InputWeights[i][j] = int16(binary.LittleEndian.Uint16(data[offset:]))
			offset += 2
		}
	}

	// l0b: [hiddenSize] i16
	for j := 0; j < hiddenSize; j++ {
		net.InputBiases[j] = int16(binary.LittleEndian.Uint16(data[offset:]))
		offset += 2
	}

	// l1w: [2*hiddenSize][buckets] i16 (already transposed by Bullet)
	// Our format: OutputWeights[bucket][2*hiddenSize] — need to transpose
	var l1wRaw [2 * hiddenSize][buckets]int16
	for i := 0; i < 2*hiddenSize; i++ {
		for b := 0; b < buckets; b++ {
			l1wRaw[i][b] = int16(binary.LittleEndian.Uint16(data[offset:]))
			offset += 2
		}
	}
	// Transpose: [concat_input][bucket] -> [bucket][concat_input]
	for b := 0; b < buckets; b++ {
		for i := 0; i < 2*hiddenSize; i++ {
			net.OutputWeights[b][i] = l1wRaw[i][b]
		}
	}

	// l1b: [buckets] i32
	for b := 0; b < buckets; b++ {
		net.OutputBias[b] = int32(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4
	}

	fmt.Printf("Parsed %d bytes of %d\n", offset, len(data))

	// Save in our v5 .nnue format
	if err := chess.SaveNNUEV5(outputPath, net); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving: %v\n", err)
		os.Exit(1)
	}

	fi, _ := os.Stat(outputPath)
	fmt.Printf("Saved %s (%d bytes, v5)\n", outputPath, fi.Size())
	fmt.Printf("Fingerprint: %s\n", net.Fingerprint())
}
