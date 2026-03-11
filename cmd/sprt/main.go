package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: sprt <coordinator|worker|create>\n")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "coordinator":
		runCoordinator(os.Args[2:])
	case "worker":
		runWorker(os.Args[2:])
	case "create":
		runCreate(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\nUsage: sprt <coordinator|worker|create>\n", os.Args[1])
		os.Exit(1)
	}
}

func runCoordinator(args []string) {
	fs := flag.NewFlagSet("coordinator", flag.ExitOnError)
	addr := fs.String("addr", ":8080", "listen address")
	repo := fs.String("repo", "", "git repo URL (for workers to clone)")
	state := fs.String("state", "state.json", "state file path")
	fs.Parse(args)

	c := NewCoordinator(*state, *repo)
	mux := http.NewServeMux()
	c.RegisterRoutes(mux)
	RegisterUI(mux)

	log.Printf("coordinator listening on %s", *addr)
	if err := http.ListenAndServe(*addr, mux); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

func runWorker(args []string) {
	fs := flag.NewFlagSet("worker", flag.ExitOnError)
	coordinator := fs.String("coordinator", "http://localhost:8080", "coordinator URL")
	repo := fs.String("repo", ".", "local repo path")
	cores := fs.Int("cores", 4, "number of CPU cores for cutechess")
	id := fs.String("id", "", "worker ID (default: hostname)")
	fs.Parse(args)

	workerID := *id
	if workerID == "" {
		h, _ := os.Hostname()
		workerID = h
	}

	w := NewWorker(workerID, *coordinator, *repo, *cores)
	w.Run()
}

func runCreate(args []string) {
	fs := flag.NewFlagSet("create", flag.ExitOnError)
	coordinator := fs.String("coordinator", "http://localhost:8080", "coordinator URL")
	id := fs.String("id", "", "experiment ID (required)")
	branch := fs.String("branch", "", "branch to test (required)")
	base := fs.String("base", "main", "base branch")
	tc := fs.String("tc", "10+0.1", "time control")
	hashMB := fs.Int("hash", 64, "hash table size in MB")
	elo0 := fs.Float64("elo0", 0, "SPRT elo0 (null hypothesis)")
	elo1 := fs.Float64("elo1", 10, "SPRT elo1 (alternative)")
	batchSize := fs.Int("batch", 50, "games per batch")
	concurrency := fs.Int("concurrency", 1, "cutechess concurrency (threads per worker)")
	nnue := fs.String("nnue", "", "NNUE file for new engine (relative to repo)")
	baseNnue := fs.String("base-nnue", "", "NNUE file for base engine (if different from -nnue)")
	fs.Parse(args)

	if *id == "" || *branch == "" {
		fmt.Fprintf(os.Stderr, "Error: -id and -branch are required\n")
		fs.PrintDefaults()
		os.Exit(1)
	}

	exp := map[string]interface{}{
		"id":          *id,
		"branch":      *branch,
		"base_branch": *base,
		"tc":          *tc,
		"hash_mb":     *hashMB,
		"batch_size":   *batchSize,
		"concurrency":  *concurrency,
		"nnue_file":      *nnue,
		"base_nnue_file": *baseNnue,
		"sprt": map[string]interface{}{
			"elo0":  *elo0,
			"elo1":  *elo1,
			"alpha": 0.05,
			"beta":  0.05,
		},
	}

	body, _ := json.Marshal(exp)
	resp, err := http.Post(*coordinator+"/api/experiments", "application/json", bytes.NewReader(body))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		var msg string
		buf := make([]byte, 1024)
		n, _ := resp.Body.Read(buf)
		msg = string(buf[:n])
		fmt.Fprintf(os.Stderr, "Error (%d): %s\n", resp.StatusCode, msg)
		os.Exit(1)
	}

	var result Experiment
	json.NewDecoder(resp.Body).Decode(&result)
	fmt.Printf("Created experiment: %s (branch=%s, base=%s, tc=%s)\n",
		result.ID, result.Branch, result.BaseBranch, result.TC)
}
