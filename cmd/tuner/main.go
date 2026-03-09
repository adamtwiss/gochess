package main

import (
	"bufio"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/signal"
	"path/filepath"
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
	case "convert":
		runConvert(os.Args[2:])
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
  convert     Convert between .dat (text) and .bin (binary) formats

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
	dataFile := fs.String("data", "training.bin", "training data file (.bin or .dat)")
	epochs := fs.Int("epochs", 500, "number of optimization epochs")
	lr := fs.Float64("lr", 1.0, "learning rate")
	l2 := fs.Float64("l2", 0, "L2 regularization strength toward initial values (0=disabled)")
	lambda := fs.Float64("lambda", 0.0, "result vs score weight: 0=score-only (default), 1=result-only")

	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: tuner tune [options]\n\nOptions:\n")
		fs.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  tuner tune -data training.bin -epochs 1000 -lr 1.0\n")
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
	isBin := strings.HasSuffix(*dataFile, ".bin")
	tbinFile := *dataFile
	if isBin {
		tbinFile = strings.TrimSuffix(tbinFile, ".bin")
	} else {
		tbinFile = strings.TrimSuffix(tbinFile, ".dat")
	}
	tbinFile += ".tbin"

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
		var err error
		if isBin {
			err = chess.PreprocessBinToFile(tuner, *dataFile, tbinFile)
		} else {
			err = chess.PreprocessToFile(tuner, *dataFile, tbinFile)
		}
		if err != nil {
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
	scoreBlend := 1.0 - *lambda
	K := tuner.TuneK(tf, scoreBlend)
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
	dataFiles := fs.String("data", "", "training data file(s), comma-separated (.bin or .dat)")
	outputFile := fs.String("output", "net.nnue", "output NNUE network file")
	epochs := fs.Int("epochs", 100, "number of training epochs")
	lr := fs.Float64("lr", 0.01, "learning rate")
	batchSize := fs.Int("batch", 16384, "batch size")
	lambda := fs.Float64("lambda", 0.0, "result vs score weight (0=score only [default], 1=result only)")
	kValue := fs.Float64("K", 400, "sigmoid scaling constant (0=auto-tune from data)")
	seed := fs.Int64("seed", 42, "random seed for weight initialization")
	positions := fs.Int("positions", 0, "limit training positions per epoch (0=use all)")
	resumeFile := fs.String("resume", "", "resume training from existing .nnue network file")

	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: tuner nnue-train [options]\n\nOptions:\n")
		fs.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  tuner nnue-train -data training.bin -epochs 100 -lr 0.001\n")
		fmt.Fprintf(os.Stderr, "  tuner nnue-train -data a.bin,b.bin -epochs 100 -lr 0.001\n")
		fmt.Fprintf(os.Stderr, "  tuner nnue-train -data training.dat -epochs 100 -lr 0.001  (legacy text format)\n")
	}
	fs.Parse(args)

	if *dataFiles == "" {
		// Try default filenames in order of preference
		if _, err := os.Stat("training.bin"); err == nil {
			*dataFiles = "training.bin"
		} else {
			*dataFiles = "training.dat"
		}
	}

	// Parse comma-separated file list
	paths := strings.Split(*dataFiles, ",")
	for i := range paths {
		paths[i] = strings.TrimSpace(paths[i])
	}

	// Detect format from first file extension
	isBinpack := strings.HasSuffix(paths[0], ".bin")

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
	}

	// Set up SIGINT handler for graceful early stop
	stopCh := make(chan struct{})
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)
	go func() {
		<-sigCh
		fmt.Fprintf(os.Stderr, "\nInterrupted — finishing current epoch and saving network...\n")
		close(stopCh)
		signal.Stop(sigCh)
	}()
	cfg.Stop = stopCh

	if isBinpack {
		runNNUETrainBinpack(trainer, paths, cfg, *outputFile, actualK, *kValue)
	} else {
		runNNUETrainLegacy(trainer, paths[0], cfg, *outputFile, actualK, *kValue, *seed)
	}
}

func runNNUETrainBinpack(trainer *chess.NNUETrainer, paths []string, cfg chess.NNUETrainConfig, outputFile string, actualK, requestedK float64) {
	bf, err := chess.OpenBinpackFiles(paths...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening binpack files: %v\n", err)
		os.Exit(1)
	}
	defer bf.Close()

	fmt.Printf("Binpack data: %d total positions from %d file(s)\n", bf.NumRecords(), len(paths))
	for _, p := range paths {
		stat, _ := os.Stat(p)
		fmt.Printf("  %s (%.1f MB, %d positions)\n", p, float64(stat.Size())/(1024*1024), stat.Size()/chess.BinpackRecordSize)
	}

	if requestedK <= 0 {
		fmt.Printf("\nTuning K (scaling constant)...\n")
		actualK = trainer.TuneKBinpack(bf, cfg.Lambda)
		cfg.K = actualK
		fmt.Printf("Using K = %.2f\n", actualK)
	} else {
		fmt.Printf("\nUsing K = %.2f (sigmoid scaling)\n", actualK)
	}

	fmt.Printf("\nTraining NNUE: epochs=%d lr=%.4f batch=%d lambda=%.2f\n",
		cfg.Epochs, cfg.LR, cfg.BatchSize, cfg.Lambda)
	fmt.Printf("%-8s  %-14s  %-14s  %s\n", "Epoch", "Train Loss", "Val Loss", "Time")
	fmt.Printf("%-8s  %-14s  %-14s  %s\n", "-----", "----------", "--------", "----")

	start := time.Now()
	epochStart := time.Now()
	trainer.TrainBinpack(bf, cfg, func(epoch int, trainLoss, valLoss float64) {
		elapsed := time.Since(epochStart)
		epochStart = time.Now()
		if epoch <= 10 || epoch%10 == 0 || epoch == cfg.Epochs {
			fmt.Printf("%-8d  %-14.8f  %-14.8f  %s\n", epoch, trainLoss, valLoss, elapsed.Round(time.Millisecond))
		}
	})

	elapsed := time.Since(start)
	fmt.Printf("\nTraining completed in %v\n", elapsed.Round(time.Second))

	// Quantize and save
	infNet := chess.QuantizeNetwork(trainer.Net)
	if err := chess.SaveNNUE(outputFile, infNet); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving network: %v\n", err)
		os.Exit(1)
	}
	fi, _ := os.Stat(outputFile)
	fmt.Printf("Network saved to %s (%.1f MB)\n", outputFile, float64(fi.Size())/(1024*1024))
}

func runNNUETrainLegacy(trainer *chess.NNUETrainer, dataFile string, cfg chess.NNUETrainConfig, outputFile string, actualK, requestedK float64, seed int64) {
	// Legacy .dat -> .nnbin path
	nnbinFile := strings.TrimSuffix(dataFile, ".dat") + ".nnbin"

	// Check if binary cache needs (re)building
	needBuild := false
	binStat, binErr := os.Stat(nnbinFile)
	if binErr != nil {
		needBuild = true
	} else {
		datStat, datErr := os.Stat(dataFile)
		if datErr == nil && datStat.ModTime().After(binStat.ModTime()) {
			needBuild = true
			fmt.Printf("Source %s is newer than %s, rebuilding cache...\n", dataFile, nnbinFile)
		}
	}

	if needBuild {
		fmt.Printf("Preprocessing %s → %s...\n", dataFile, nnbinFile)
		start := time.Now()
		numTrain, numVal, err := chess.PreprocessNNUEToFile(dataFile, nnbinFile)
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

	bf, err := chess.OpenNNBinFile(nnbinFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening binary file: %v\n", err)
		os.Exit(1)
	}
	defer bf.Close()
	fmt.Printf("Training data: %d train + %d validation positions\n", bf.NumTrain, bf.NumValidation)

	if requestedK <= 0 {
		fmt.Printf("\nUsing default K = %.2f (centipawn scale)\n", actualK)
	} else {
		fmt.Printf("\nUsing K = %.2f (sigmoid scaling)\n", actualK)
	}

	fmt.Printf("\nTraining NNUE: epochs=%d lr=%.4f batch=%d lambda=%.2f\n",
		cfg.Epochs, cfg.LR, cfg.BatchSize, cfg.Lambda)
	fmt.Printf("%-8s  %-14s  %-14s  %s\n", "Epoch", "Train Loss", "Val Loss", "Time")
	fmt.Printf("%-8s  %-14s  %-14s  %s\n", "-----", "----------", "--------", "----")

	start := time.Now()
	epochStart := time.Now()
	trainer.Train(bf, cfg, func(epoch int, trainLoss, valLoss float64) {
		elapsed := time.Since(epochStart)
		epochStart = time.Now()
		if epoch <= 10 || epoch%10 == 0 || epoch == cfg.Epochs {
			fmt.Printf("%-8d  %-14.8f  %-14.8f  %s\n", epoch, trainLoss, valLoss, elapsed.Round(time.Millisecond))
		}
	})

	elapsed := time.Since(start)
	fmt.Printf("\nTraining completed in %v\n", elapsed.Round(time.Second))

	infNet := chess.QuantizeNetwork(trainer.Net)
	if err := chess.SaveNNUE(outputFile, infNet); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving network: %v\n", err)
		os.Exit(1)
	}
	fi, _ := os.Stat(outputFile)
	fmt.Printf("Network saved to %s (%.1f MB)\n", outputFile, float64(fi.Size())/(1024*1024))
}

func runConvert(args []string) {
	fs := flag.NewFlagSet("convert", flag.ExitOnError)
	from := fs.String("from", "", "source file (.dat or .bin)")
	to := fs.String("to", "", "destination file (.dat or .bin)")
	shuffle := fs.Bool("shuffle", false, "shuffle records (only for .dat -> .bin, uses seed 42)")

	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: tuner convert [options]\n\nOptions:\n")
		fs.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  tuner convert -from training.dat -to training.bin\n")
		fmt.Fprintf(os.Stderr, "  tuner convert -from training.bin -to training.dat\n")
	}
	fs.Parse(args)

	if *from == "" || *to == "" {
		fs.Usage()
		os.Exit(1)
	}

	fromExt := strings.ToLower(filepath.Ext(*from))
	toExt := strings.ToLower(filepath.Ext(*to))

	start := time.Now()

	switch {
	case fromExt == ".dat" && toExt == ".bin":
		fmt.Printf("Converting %s → %s...\n", *from, *to)
		_ = *shuffle // ConvertDatToBinpack always shuffles
		count, err := chess.ConvertDatToBinpack(*from, *to)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		stat, _ := os.Stat(*to)
		fmt.Printf("Done: %d positions in %v (%.1f MB)\n", count, time.Since(start).Round(time.Millisecond), float64(stat.Size())/(1024*1024))

	case fromExt == ".bin" && toExt == ".dat":
		fmt.Printf("Converting %s → %s...\n", *from, *to)
		count, err := chess.ConvertBinpackToDat(*from, *to)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Done: %d positions in %v\n", count, time.Since(start).Round(time.Millisecond))

	case fromExt == ".bin" && toExt == ".bin":
		// Concatenation / shuffle of multiple binpack files
		fmt.Printf("Copying %s → %s...\n", *from, *to)
		if *shuffle {
			// Read all, shuffle, write
			bf, err := chess.OpenBinpackFiles(*from)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			defer bf.Close()
			n := bf.NumRecords()
			indices := make([]int, n)
			for i := range indices {
				indices[i] = i
			}
			rng := rand.New(rand.NewSource(42))
			rng.Shuffle(n, func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })

			out, err := os.Create(*to)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			defer out.Close()
			for _, idx := range indices {
				rec, err := bf.ReadRecord(idx)
				if err != nil {
					fmt.Fprintf(os.Stderr, "Error reading record %d: %v\n", idx, err)
					os.Exit(1)
				}
				out.Write(rec[:])
			}
			fmt.Printf("Done: %d positions shuffled in %v\n", n, time.Since(start).Round(time.Millisecond))
		} else {
			// Just copy
			data, err := os.ReadFile(*from)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			if err := os.WriteFile(*to, data, 0644); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("Done: %d positions copied in %v\n", len(data)/chess.BinpackRecordSize, time.Since(start).Round(time.Millisecond))
		}

	default:
		fmt.Fprintf(os.Stderr, "Unsupported conversion: %s → %s\n", fromExt, toExt)
		fmt.Fprintf(os.Stderr, "Supported: .dat → .bin, .bin → .dat, .bin → .bin (with -shuffle)\n")
		os.Exit(1)
	}
}
