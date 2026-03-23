package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Worker polls the coordinator for work and runs cutechess-cli batches.
type Worker struct {
	id              string
	coordinatorURL  string
	repoDir         string
	cores           int
	cacheDir        string
	openingsFile    string
	pollInterval    time.Duration
	cutechessPath   string
	engineRegistry  map[string]string // name -> absolute path to rival engine binary
}

// NewWorker creates a worker instance.
func NewWorker(id, coordinatorURL, repoDir string, cores int) *Worker {
	home, _ := os.UserHomeDir()
	cacheDir := filepath.Join(home, ".sprt-cache")
	os.MkdirAll(cacheDir, 0755)

	// Find cutechess-cli
	cutechess := "cutechess-cli"
	if p, err := exec.LookPath("cutechess-cli"); err == nil {
		cutechess = p
	}

	// Resolve repoDir to absolute path at startup so NNUE paths
	// remain valid even if cutechess changes the working directory.
	absRepoDir, err := filepath.Abs(repoDir)
	if err != nil {
		absRepoDir = repoDir
	}

	return &Worker{
		id:             id,
		coordinatorURL: strings.TrimRight(coordinatorURL, "/"),
		repoDir:        absRepoDir,
		cores:          cores,
		cacheDir:       cacheDir,
		pollInterval:   10 * time.Second,
		cutechessPath:  cutechess,
		engineRegistry: make(map[string]string),
	}
}

// LoadEngines reads a JSON engine registry file mapping engine names to binary paths.
// Format: {"Texel": "/path/to/texel", "Ethereal": "/path/to/ethereal", ...}
func (w *Worker) LoadEngines(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("reading engines file: %w", err)
	}
	var registry map[string]string
	if err := json.Unmarshal(data, &registry); err != nil {
		return fmt.Errorf("parsing engines file: %w", err)
	}
	// Validate all paths exist and resolve to absolute
	for name, binPath := range registry {
		abs, err := filepath.Abs(binPath)
		if err != nil {
			return fmt.Errorf("engine %s: invalid path %q: %w", name, binPath, err)
		}
		if _, err := os.Stat(abs); err != nil {
			return fmt.Errorf("engine %s: binary not found at %s", name, abs)
		}
		w.engineRegistry[name] = abs
		log.Printf("registered engine: %s -> %s", name, abs)
	}
	return nil
}

// availableEngineNames returns the names of all registered rival engines.
func (w *Worker) availableEngineNames() []string {
	names := make([]string, 0, len(w.engineRegistry))
	for name := range w.engineRegistry {
		names = append(names, name)
	}
	return names
}

// Run starts the worker polling loop.
func (w *Worker) Run() {
	log.Printf("worker %s starting: coordinator=%s repo=%s cores=%d",
		w.id, w.coordinatorURL, w.repoDir, w.cores)

	consecutiveErrors := 0

	for {
		claim, err := w.claimWork()
		if err != nil {
			log.Printf("error claiming work: %v", err)
			time.Sleep(w.pollInterval)
			continue
		}
		if claim == nil {
			consecutiveErrors = 0 // reset on idle (no work available is not an error)
			time.Sleep(w.pollInterval)
			continue
		}

		log.Printf("claimed work: experiment=%s branch=%s batch_size=%d opening_start=%d",
			claim.ExperimentID, claim.Branch, claim.BatchSize, claim.OpeningStart)

		report := w.runBatch(claim)
		if err := w.reportResult(report); err != nil {
			log.Printf("error reporting result: %v", err)
		}

		if report.Error != "" {
			consecutiveErrors++
			backoff := time.Duration(consecutiveErrors) * 30 * time.Second
			if backoff > 5*time.Minute {
				backoff = 5 * time.Minute
			}
			log.Printf("batch error (%d consecutive), backing off %v", consecutiveErrors, backoff)
			time.Sleep(backoff)
		} else {
			consecutiveErrors = 0
		}
	}
}

func (w *Worker) claimWork() (*WorkClaim, error) {
	body, _ := json.Marshal(ClaimRequest{
		WorkerID:         w.id,
		Cores:            w.cores,
		AvailableEngines: w.availableEngineNames(),
	})
	resp, err := http.Post(w.coordinatorURL+"/api/work/claim", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNoContent {
		return nil, nil // no work
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status: %d", resp.StatusCode)
	}

	var claim WorkClaim
	if err := json.NewDecoder(resp.Body).Decode(&claim); err != nil {
		return nil, err
	}
	return &claim, nil
}

func (w *Worker) reportResult(report BatchReport) error {
	body, _ := json.Marshal(report)
	resp, err := http.Post(w.coordinatorURL+"/api/work/report", "application/json", bytes.NewReader(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("report failed with status: %d", resp.StatusCode)
	}

	var result SPRTResult
	json.NewDecoder(resp.Body).Decode(&result)
	total := result.Wins + result.Draws + result.Losses
	log.Printf("reported: total=%d Elo=%.1f±%.1f LLR=%.2f [%s]",
		total, result.Elo, result.EloErr, result.LLR, result.Status)
	return nil
}

func (w *Worker) runBatch(claim *WorkClaim) BatchReport {
	report := BatchReport{
		ExperimentID: claim.ExperimentID,
		WorkerID:     w.id,
	}

	// Fetch latest from remote
	if err := w.gitFetch(); err != nil {
		report.Error = fmt.Sprintf("git fetch failed: %v", err)
		log.Printf("error: %s", report.Error)
		return report
	}

	// Build GoChess binary (for gauntlet, only need the variant; for SPRT, need both)
	newBin, err := w.buildBranch(claim.Branch)
	if err != nil {
		report.Error = fmt.Sprintf("build %s failed: %v", claim.Branch, err)
		log.Printf("error: %s", report.Error)
		return report
	}
	var baseBin string
	if claim.Mode != "gauntlet" {
		baseBin, err = w.buildBranch(claim.BaseBranch)
		if err != nil {
			report.Error = fmt.Sprintf("build %s failed: %v", claim.BaseBranch, err)
			log.Printf("error: %s", report.Error)
			return report
		}
	}

	// Ensure NNUE file exists (run fetch-net if needed)
	if claim.NNUEFile != "" {
		nnuePath := w.resolveNNUEPath(claim.NNUEFile)
		if _, err := os.Stat(nnuePath); os.IsNotExist(err) {
			log.Printf("NNUE file %s not found, running fetch-net...", nnuePath)
			cmd := exec.Command(newBin, "fetch-net")
			cmd.Dir = w.repoDir
			if out, err := cmd.CombinedOutput(); err != nil {
				log.Printf("fetch-net output: %s", string(out))
				// Not fatal — the engine might work via UCI option
			}
		}
	}

	// Find openings file
	openingsFile := w.findOpeningsFile()
	if openingsFile == "" {
		report.Error = "no openings file found (testdata/noob_3moves.epd)"
		log.Printf("error: %s", report.Error)
		return report
	}

	// Build cutechess command
	var args []string
	if claim.Mode == "gauntlet" {
		opponentBin, ok := w.engineRegistry[claim.OpponentName]
		if !ok {
			report.Error = fmt.Sprintf("opponent engine %q not in registry", claim.OpponentName)
			log.Printf("error: %s", report.Error)
			return report
		}
		report.OpponentName = claim.OpponentName
		args = w.buildGauntletCutechessArgs(claim, newBin, opponentBin, openingsFile)
	} else {
		args = w.buildCutechessArgs(claim, newBin, baseBin, openingsFile)
	}
	log.Printf("running: %s %s", w.cutechessPath, strings.Join(args, " "))

	cmd := exec.Command(w.cutechessPath, args...)
	cmd.Dir = w.repoDir
	output, err := cmd.CombinedOutput()
	if err != nil {
		report.Error = fmt.Sprintf("cutechess failed: %v\n%s", err, lastLines(string(output), 10))
		log.Printf("error: %s", report.Error)
		return report
	}

	// Parse results
	wins, draws, losses, parseErr := parseCutechessOutput(string(output))
	if parseErr != nil {
		report.Error = fmt.Sprintf("parse error: %v\n%s", parseErr, lastLines(string(output), 10))
		log.Printf("error: %s", report.Error)
		return report
	}

	report.Wins = wins
	report.Draws = draws
	report.Losses = losses
	log.Printf("batch complete: +%d =%d -%d", wins, draws, losses)
	return report
}

func (w *Worker) gitFetch() error {
	cmd := exec.Command("git", "fetch", "origin", "--prune")
	cmd.Dir = w.repoDir
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%v: %s", err, output)
	}
	return nil
}

// resolveBranch resolves a branch name to a commit hash, trying multiple strategies.
func (w *Worker) resolveBranch(branch string) (string, error) {
	// Try origin/<branch> first (most common for remote branches)
	candidates := []string{
		"origin/" + branch,
		branch,
		"refs/heads/" + branch,
		"refs/tags/" + branch,
	}

	var lastErr error
	for _, ref := range candidates {
		cmd := exec.Command("git", "rev-parse", "--verify", ref)
		cmd.Dir = w.repoDir
		out, err := cmd.Output()
		if err == nil {
			return strings.TrimSpace(string(out)), nil
		}
		lastErr = err
	}

	// Try fetching the specific branch in case refspec didn't cover it
	cmd := exec.Command("git", "fetch", "origin", branch)
	cmd.Dir = w.repoDir
	if fetchOut, err := cmd.CombinedOutput(); err == nil {
		cmd = exec.Command("git", "rev-parse", "--verify", "origin/"+branch)
		cmd.Dir = w.repoDir
		out, err := cmd.Output()
		if err == nil {
			return strings.TrimSpace(string(out)), nil
		}
	} else {
		log.Printf("fetch origin %s failed: %s", branch, strings.TrimSpace(string(fetchOut)))
	}

	// List available branches for diagnostics
	cmd = exec.Command("git", "branch", "-r", "--list", "origin/*"+branch+"*")
	cmd.Dir = w.repoDir
	if listOut, err := cmd.Output(); err == nil && len(listOut) > 0 {
		log.Printf("similar remote branches: %s", strings.TrimSpace(string(listOut)))
	}

	return "", fmt.Errorf("cannot resolve ref %q (tried %v): %v", branch, candidates, lastErr)
}

// buildBranch builds the chess binary for a given branch, using a commit-hash cache.
func (w *Worker) buildBranch(branch string) (string, error) {
	hash, err := w.resolveBranch(branch)
	if err != nil {
		return "", err
	}

	// Check cache
	binPath := filepath.Join(w.cacheDir, hash, "chess")
	if _, err := os.Stat(binPath); err == nil {
		log.Printf("using cached binary for %s (%s)", branch, hash[:8])
		return binPath, nil
	}

	// Build in a temporary worktree
	worktree := filepath.Join(w.cacheDir, "worktree-"+hash[:12])
	defer os.RemoveAll(worktree)

	cmd := exec.Command("git", "worktree", "add", "--detach", worktree, hash)
	cmd.Dir = w.repoDir
	if out, err := cmd.CombinedOutput(); err != nil {
		// Worktree might already exist from a previous failed build
		os.RemoveAll(worktree)
		cmd = exec.Command("git", "worktree", "add", "--detach", worktree, hash)
		cmd.Dir = w.repoDir
		if out, err = cmd.CombinedOutput(); err != nil {
			return "", fmt.Errorf("worktree add: %v: %s", err, out)
		}
	}

	binDir := filepath.Join(w.cacheDir, hash)
	os.MkdirAll(binDir, 0755)

	cmd = exec.Command("go", "build", "-o", binPath, "./cmd/chess")
	cmd.Dir = worktree
	if out, err := cmd.CombinedOutput(); err != nil {
		return "", fmt.Errorf("go build: %v: %s", err, out)
	}

	// Clean up worktree
	cmd = exec.Command("git", "worktree", "remove", "--force", worktree)
	cmd.Dir = w.repoDir
	cmd.Run() // best-effort

	log.Printf("built %s (%s)", branch, hash[:8])
	return binPath, nil
}

func (w *Worker) findOpeningsFile() string {
	// Check repo's testdata
	epd := filepath.Join(w.repoDir, "testdata", "noob_3moves.epd")
	if _, err := os.Stat(epd); err == nil {
		return epd
	}
	return ""
}

func (w *Worker) resolveNNUEPath(path string) string {
	if path == "" {
		return ""
	}
	if filepath.IsAbs(path) {
		return path
	}
	// Always return absolute path so cutechess engines can find the file
	abs, err := filepath.Abs(filepath.Join(w.repoDir, path))
	if err != nil {
		return filepath.Join(w.repoDir, path)
	}
	return abs
}

func (w *Worker) buildCutechessArgs(claim *WorkClaim, newBin, baseBin, openingsFile string) []string {
	newNNUE := w.resolveNNUEPath(claim.NNUEFile)
	baseNNUE := w.resolveNNUEPath(claim.BaseNNUEFile)
	if baseNNUE == "" {
		baseNNUE = newNNUE // same net for both if base not specified
	}

	// When engines use different NNUE files, set options per-engine.
	// When same, use -each for shared options.
	perEngineNNUE := newNNUE != baseNNUE && newNNUE != "" && baseNNUE != ""

	// Engine definitions
	newEngineArgs := []string{"-engine", "name=New", "cmd=" + newBin, "proto=uci"}
	if perEngineNNUE {
		newEngineArgs = append(newEngineArgs, "option.UseNNUE=true", "option.NNUEFile="+newNNUE)
	}

	baseEngineArgs := []string{"-engine", "name=Base", "cmd=" + baseBin, "proto=uci"}
	if perEngineNNUE {
		baseEngineArgs = append(baseEngineArgs, "option.UseNNUE=true", "option.NNUEFile="+baseNNUE)
	}

	args := []string{"-tournament", "gauntlet"}
	args = append(args, newEngineArgs...)
	args = append(args, baseEngineArgs...)

	// Shared options via -each
	args = append(args, "-each",
		fmt.Sprintf("tc=0/%s", claim.TC),
		fmt.Sprintf("option.Hash=%d", claim.HashMB),
		"option.OwnBook=false",
		"option.MoveOverhead=100",
	)

	// Shared NNUE (same file for both engines)
	if !perEngineNNUE && newNNUE != "" {
		args = append(args, "option.UseNNUE=true", "option.NNUEFile="+newNNUE)
	}

	for k, v := range claim.Options {
		args = append(args, fmt.Sprintf("option.%s=%s", k, v))
	}

	// Use local core count for concurrency (cores/2 since each game needs 2 engines)
	// Fall back to experiment-specified concurrency if cores not set
	concurrency := claim.Concurrency
	if w.cores >= 2 {
		concurrency = w.cores / 2
		if concurrency < 1 {
			concurrency = 1
		}
	}

	args = append(args,
		"-rounds", strconv.Itoa(claim.BatchSize),
		"-concurrency", strconv.Itoa(concurrency),
		"-openings", "file="+openingsFile, "format=epd", "order=random",
		fmt.Sprintf("start=%d", claim.OpeningStart+1), // cutechess uses 1-based
		"-draw", "movenumber=20", "movecount=10", "score=10",
		"-resign", "movecount=3", "score=500", "twosided=true",
		"-recover",
	)

	return args
}

// buildGauntletCutechessArgs constructs cutechess-cli arguments for gauntlet mode.
// GoChess plays against a single rival engine per batch.
func (w *Worker) buildGauntletCutechessArgs(claim *WorkClaim, gochessBin, opponentBin, openingsFile string) []string {
	nnue := w.resolveNNUEPath(claim.NNUEFile)

	// GoChess engine
	goChessArgs := []string{"-engine", "name=GoChess", "cmd=" + gochessBin, "proto=uci",
		"option.OwnBook=false", "option.MoveOverhead=100"}
	if nnue != "" {
		goChessArgs = append(goChessArgs, "option.UseNNUE=true", "option.NNUEFile="+nnue)
	}

	// Rival engine — minimal options (just hash via -each)
	rivalArgs := []string{"-engine", "name=" + claim.OpponentName, "cmd=" + opponentBin, "proto=uci"}

	args := []string{"-tournament", "gauntlet"}
	args = append(args, goChessArgs...)
	args = append(args, rivalArgs...)

	// Shared options
	args = append(args, "-each",
		fmt.Sprintf("tc=0/%s", claim.TC),
		fmt.Sprintf("option.Hash=%d", claim.HashMB),
	)

	for k, v := range claim.Options {
		args = append(args, fmt.Sprintf("option.%s=%s", k, v))
	}

	// Use local core count for concurrency
	concurrency := claim.Concurrency
	if w.cores >= 2 {
		concurrency = w.cores / 2
		if concurrency < 1 {
			concurrency = 1
		}
	}

	args = append(args,
		"-rounds", strconv.Itoa(claim.BatchSize),
		"-concurrency", strconv.Itoa(concurrency),
		"-openings", "file="+openingsFile, "format=epd", "order=random",
		"-draw", "movenumber=20", "movecount=10", "score=10",
		"-resign", "movecount=3", "score=500", "twosided=true",
		"-recover",
	)

	return args
}

// parseCutechessOutput extracts W/D/L from cutechess-cli output.
// Looks for line: "Score of New vs Base: W - L - D  [pct] N"
var scoreRe = regexp.MustCompile(`Score of .+ vs .+: (\d+) - (\d+) - (\d+)`)

func parseCutechessOutput(output string) (wins, draws, losses int, err error) {
	lines := strings.Split(output, "\n")
	// Search from the end for the Score line
	for i := len(lines) - 1; i >= 0; i-- {
		m := scoreRe.FindStringSubmatch(lines[i])
		if m != nil {
			w, _ := strconv.Atoi(m[1])
			l, _ := strconv.Atoi(m[2])
			d, _ := strconv.Atoi(m[3])
			return w, d, l, nil
		}
	}
	return 0, 0, 0, fmt.Errorf("no score line found in cutechess output")
}

func lastLines(s string, n int) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) <= n {
		return s
	}
	return strings.Join(lines[len(lines)-n:], "\n")
}
