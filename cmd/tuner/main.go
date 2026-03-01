package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"
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
	default:
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Fprintf(os.Stderr, `Usage: tuner <command> [options]

Commands:
  selfplay    Generate training data from self-play games
  tune        Optimize evaluation parameters from training data
  nnue-train  Train an NNUE network from training data

Run 'tuner <command> -h' for command-specific options.
`)
}

func runSelfPlay(args []string) {
	fs := flag.NewFlagSet("selfplay", flag.ExitOnError)
	games := fs.Int("games", 1000, "number of games to play")
	timeMS := fs.Int("time", 200, "time per move in milliseconds")
	depth := fs.Int("depth", 64, "max search depth per move")
	threads := fs.Int("threads", 1, "search threads per game (Lazy SMP)")
	concurrency := fs.Int("concurrency", 1, "number of games to play concurrently")
	openings := fs.String("openings", "testdata/noob_3moves.epd", "EPD file with opening positions")
	output := fs.String("output", "training.dat", "output file for training data")
	hashMB := fs.Int("hash", 16, "TT size in MB per game")

	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: tuner selfplay [options]\n\nOptions:\n")
		fs.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExample:\n")
		fmt.Fprintf(os.Stderr, "  tuner selfplay -games 20000 -time 200 -concurrency 6 -output training.dat\n")
	}
	fs.Parse(args)

	cfg := chess.SelfPlayConfig{
		TimePerMove:  time.Duration(*timeMS) * time.Millisecond,
		MaxDepth:     *depth,
		NumGames:     *games,
		Threads:      *threads,
		Concurrency:  *concurrency,
		OpeningsFile: *openings,
		OutputFile:   *output,
		HashMB: *hashMB,
	}

	fmt.Printf("Self-play configuration:\n")
	fmt.Printf("  Games:       %d\n", cfg.NumGames)
	fmt.Printf("  Time/move:   %v\n", cfg.TimePerMove)
	fmt.Printf("  Max depth:   %d\n", cfg.MaxDepth)
	fmt.Printf("  Threads:     %d (per game)\n", cfg.Threads)
	fmt.Printf("  Concurrency: %d (parallel games)\n", cfg.Concurrency)
	fmt.Printf("  Openings:    %s\n", cfg.OpeningsFile)
	fmt.Printf("  Output:      %s\n", cfg.OutputFile)
	fmt.Printf("  Hash:        %d MB (per game)\n", cfg.HashMB)
	if stat, err := os.Stat(cfg.OutputFile); err == nil {
		fmt.Printf("  (appending to existing file, %d bytes)\n", stat.Size())
	}
	fmt.Println()

	start := time.Now()
	totalPositions := 0

	err := chess.RunSelfPlay(cfg, func(gameNum int, game chess.SelfPlayGame) {
		totalPositions += len(game.Positions)
		elapsed := time.Since(start)
		gps := float64(gameNum) / elapsed.Seconds()
		fmt.Printf("\rGame %d/%d (%s, %d plies, %d positions) [%.1f games/s, %d total positions]        ",
			gameNum, cfg.NumGames, game.ResultStr, game.Plies, len(game.Positions),
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
	dataFile := fs.String("data", "training.dat", "training data file")
	epochs := fs.Int("epochs", 500, "number of optimization epochs")
	lr := fs.Float64("lr", 1.0, "learning rate")
	lambda := fs.Float64("lambda", 0, "L2 regularization strength toward initial values (0=disabled)")
	scoreBlend := fs.Float64("score-blend", 0, "blend search scores into loss: 0=result-only (default), 1=score-only")

	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: tuner tune [options]\n\nOptions:\n")
		fs.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExample:\n")
		fmt.Fprintf(os.Stderr, "  tuner tune -data training.dat -epochs 1000 -lr 1.0\n")
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
	tbinFile := strings.TrimSuffix(*dataFile, ".dat") + ".tbin"

	// Check if .tbin needs to be (re)built
	needBuild := false
	tbinStat, tbinErr := os.Stat(tbinFile)
	if tbinErr != nil {
		needBuild = true
	} else {
		datStat, datErr := os.Stat(*dataFile)
		if datErr == nil && datStat.ModTime().After(tbinStat.ModTime()) {
			needBuild = true
			fmt.Printf("Source %s is newer than %s, rebuilding cache...\n", *dataFile, tbinFile)
		}
	}

	if needBuild {
		fmt.Printf("Preprocessing %s → %s...\n", *dataFile, tbinFile)
		start := time.Now()
		if err := chess.PreprocessToFile(tuner, *dataFile, tbinFile); err != nil {
			fmt.Fprintf(os.Stderr, "Error preprocessing: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Preprocessing done in %v\n", time.Since(start).Round(time.Millisecond))
	}

	// Open trace file (mmap)
	tf, err := chess.OpenTraceFile(tbinFile, tuner.NumParams())
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

	// Tune K
	fmt.Printf("\nTuning K (scaling constant)...\n")
	K := tuner.TuneK(tf, *scoreBlend)
	initialError := tuner.ComputeTrainError(tf, K, *scoreBlend)
	initialValError := tuner.ComputeValidationError(tf, K, *scoreBlend)
	fmt.Printf("Optimal K = %.2f, initial train error = %.8f, val error = %.8f\n\n", K, initialError, initialValError)

	// Run optimizer
	cfg := chess.DefaultTuneConfig()
	cfg.Epochs = *epochs
	cfg.LR = *lr
	cfg.Lambda = *lambda
	cfg.ScoreBlend = *scoreBlend

	fmt.Printf("Running Adam optimizer: epochs=%d, lr=%.2f, lambda=%.1e\n", cfg.Epochs, cfg.LR, cfg.Lambda)
	fmt.Printf("%-8s  %-14s  %-14s\n", "Epoch", "Train Error", "Val Error")
	fmt.Printf("%-8s  %-14s  %-14s\n", "-----", "-----------", "---------")

	tuner.Tune(tf, K, cfg, func(epoch int, trainErr, valErr float64) {
		if epoch <= 10 || epoch%10 == 0 || epoch == cfg.Epochs {
			fmt.Printf("%-8d  %.8f    %.8f\n", epoch, trainErr, valErr)
		}
	})

	// Print results
	fmt.Printf("\n=== Tuned Parameters ===\n\n")
	w := bufio.NewWriter(os.Stdout)
	tuner.PrintParams(w)
	w.Flush()

	finalError := tuner.ComputeTrainError(tf, K, *scoreBlend)
	finalValError := tuner.ComputeValidationError(tf, K, *scoreBlend)
	fmt.Printf("\nTrain:      initial=%.8f  final=%.8f  improvement=%.4f%%\n",
		initialError, finalError, (initialError-finalError)/initialError*100)
	fmt.Printf("Validation: initial=%.8f  final=%.8f  improvement=%.4f%%\n",
		initialValError, finalValError, (initialValError-finalValError)/initialValError*100)
}

func runNNUETrain(args []string) {
	fs := flag.NewFlagSet("nnue-train", flag.ExitOnError)
	dataFile := fs.String("data", "training.dat", "training data file (FEN;result or FEN;score;result)")
	outputFile := fs.String("output", "net.nnue", "output NNUE network file")
	epochs := fs.Int("epochs", 100, "number of training epochs")
	lr := fs.Float64("lr", 0.001, "learning rate")
	batchSize := fs.Int("batch", 16384, "batch size")
	lambda := fs.Float64("lambda", 0.5, "result vs score weight (0=score only, 1=result only)")
	kValue := fs.Float64("K", 400, "sigmoid scaling constant (default 400)")
	seed := fs.Int64("seed", 42, "random seed for weight initialization")
	positions := fs.Int("positions", 0, "limit training positions per epoch (0=use all)")
	resumeFile := fs.String("resume", "", "resume training from existing .nnue network file")

	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: tuner nnue-train [options]\n\nOptions:\n")
		fs.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExample:\n")
		fmt.Fprintf(os.Stderr, "  tuner nnue-train -data training.dat -epochs 100 -lr 0.001 -output net.nnue\n")
	}
	fs.Parse(args)

	// Derive .nnbin filename
	nnbinFile := strings.TrimSuffix(*dataFile, ".dat") + ".nnbin"

	// Check if binary cache needs (re)building
	needBuild := false
	binStat, binErr := os.Stat(nnbinFile)
	if binErr != nil {
		needBuild = true
	} else {
		datStat, datErr := os.Stat(*dataFile)
		if datErr == nil && datStat.ModTime().After(binStat.ModTime()) {
			needBuild = true
			fmt.Printf("Source %s is newer than %s, rebuilding cache...\n", *dataFile, nnbinFile)
		}
	}

	if needBuild {
		fmt.Printf("Preprocessing %s → %s...\n", *dataFile, nnbinFile)
		start := time.Now()
		numTrain, numVal, err := chess.PreprocessNNUEToFile(*dataFile, nnbinFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error preprocessing: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Preprocessing done: %d train + %d validation in %v\n",
			numTrain, numVal, time.Since(start).Round(time.Millisecond))
	} else {
		fmt.Printf("Using cached binary file: %s (%.1f MB)\n", nnbinFile,
			float64(binStat.Size())/(1024*1024))
	}

	// Open binary file
	bf, err := chess.OpenNNBinFile(nnbinFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening binary file: %v\n", err)
		os.Exit(1)
	}
	defer bf.Close()
	fmt.Printf("Training data: %d train + %d validation positions\n", bf.NumTrain, bf.NumValidation)

	// Create trainer
	trainer := chess.NewNNUETrainer(*seed)

	// Resume from existing network if specified
	if *resumeFile != "" {
		net, err := chess.LoadNNUE(*resumeFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading resume network: %v\n", err)
			os.Exit(1)
		}
		trainer.LoadWeights(net)
		fmt.Printf("Resumed weights from %s\n", *resumeFile)
	}

	cfg := chess.NNUETrainConfig{
		Epochs:       *epochs,
		LR:           *lr,
		BatchSize:    *batchSize,
		Lambda:       *lambda,
		K:            *kValue,
		MaxPositions: *positions,
	}

	fmt.Printf("\nUsing K = %.2f (sigmoid scaling)\n\n", cfg.K)

	fmt.Printf("Training NNUE: epochs=%d lr=%.4f batch=%d lambda=%.2f\n",
		cfg.Epochs, cfg.LR, cfg.BatchSize, cfg.Lambda)
	fmt.Printf("%-8s  %-14s  %-14s\n", "Epoch", "Train Loss", "Val Loss")
	fmt.Printf("%-8s  %-14s  %-14s\n", "-----", "----------", "--------")

	start := time.Now()
	trainer.Train(bf, cfg, func(epoch int, trainLoss, valLoss float64) {
		if epoch <= 10 || epoch%10 == 0 || epoch == cfg.Epochs {
			fmt.Printf("%-8d  %.8f    %.8f\n", epoch, trainLoss, valLoss)
		}
	})

	elapsed := time.Since(start)
	fmt.Printf("\nTraining completed in %v\n", elapsed.Round(time.Second))

	// Quantize and save
	infNet := chess.QuantizeNetwork(trainer.Net)
	if err := chess.SaveNNUE(*outputFile, infNet); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving network: %v\n", err)
		os.Exit(1)
	}
	fi, _ := os.Stat(*outputFile)
	fmt.Printf("Network saved to %s (%.1f MB)\n", *outputFile, float64(fi.Size())/(1024*1024))
}
