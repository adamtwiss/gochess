package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// Experiment represents a single SPRT test.
type Experiment struct {
	ID          string            `json:"id"`
	Branch      string            `json:"branch"`
	BaseBranch  string            `json:"base_branch"`
	SPRT        SPRTConfig        `json:"sprt"`
	TC          string            `json:"tc"`
	HashMB      int               `json:"hash_mb"`
	Concurrency int               `json:"concurrency"` // threads per worker
	BatchSize   int               `json:"batch_size"`
	NNUEFile    string            `json:"nnue_file"`
	BaseNNUEFile string           `json:"base_nnue_file,omitempty"` // if different from NNUEFile
	Options     map[string]string `json:"options"`
	CreatedAt        time.Time         `json:"created_at"`
	Result           SPRTResult        `json:"result"`
	NextOpening      int               `json:"next_opening"`
	Batches          []BatchRecord     `json:"batches"`
	ConsecutiveErrors int              `json:"consecutive_errors"`
}

// BatchRecord stores a completed batch report.
type BatchRecord struct {
	WorkerID  string    `json:"worker_id"`
	Wins      int       `json:"wins"`
	Draws     int       `json:"draws"`
	Losses    int       `json:"losses"`
	Error     string    `json:"error,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}

// PendingClaim tracks an outstanding work claim.
type PendingClaim struct {
	ExperimentID string
	WorkerID     string
	OpeningStart int
	BatchSize    int
	ClaimedAt    time.Time
}

// WorkClaim is sent to a worker when it claims work.
type WorkClaim struct {
	ExperimentID string            `json:"experiment_id"`
	Branch       string            `json:"branch"`
	BaseBranch   string            `json:"base_branch"`
	TC           string            `json:"tc"`
	HashMB       int               `json:"hash_mb"`
	Concurrency  int               `json:"concurrency"`
	BatchSize    int               `json:"batch_size"`
	OpeningStart int               `json:"opening_start"`
	RepoURL      string            `json:"repo_url"`
	NNUEFile     string            `json:"nnue_file"`
	BaseNNUEFile string            `json:"base_nnue_file,omitempty"`
	Options      map[string]string `json:"options"`
}

// BatchReport is sent by a worker after completing a batch.
type BatchReport struct {
	ExperimentID string `json:"experiment_id"`
	WorkerID     string `json:"worker_id"`
	Wins         int    `json:"wins"`
	Draws        int    `json:"draws"`
	Losses       int    `json:"losses"`
	Error        string `json:"error,omitempty"`
}

// ClaimRequest is sent by a worker to request work.
type ClaimRequest struct {
	WorkerID string `json:"worker_id"`
	Cores    int    `json:"cores"`
}

// Coordinator manages experiments and distributes work.
type Coordinator struct {
	mu            sync.Mutex
	experiments   []*Experiment
	pendingClaims []PendingClaim
	stateFile     string
	repoURL       string
	staleTimeout  time.Duration
}

// NewCoordinator creates a coordinator, loading state from disk if available.
func NewCoordinator(stateFile, repoURL string) *Coordinator {
	c := &Coordinator{
		stateFile:    stateFile,
		repoURL:      repoURL,
		staleTimeout: 30 * time.Minute,
	}
	c.loadState()
	go c.staleBatchReaper()
	return c
}

// staleBatchReaper periodically reclaims stale work claims.
func (c *Coordinator) staleBatchReaper() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	for range ticker.C {
		c.mu.Lock()
		now := time.Now()
		var kept []PendingClaim
		for _, pc := range c.pendingClaims {
			if now.Sub(pc.ClaimedAt) < c.staleTimeout {
				kept = append(kept, pc)
			} else {
				log.Printf("reclaiming stale batch: experiment=%s worker=%s openings=%d",
					pc.ExperimentID, pc.WorkerID, pc.OpeningStart)
			}
		}
		c.pendingClaims = kept
		c.mu.Unlock()
	}
}

type stateData struct {
	Experiments []*Experiment `json:"experiments"`
}

func (c *Coordinator) loadState() {
	data, err := os.ReadFile(c.stateFile)
	if err != nil {
		return // no state file, start fresh
	}
	var st stateData
	if err := json.Unmarshal(data, &st); err != nil {
		log.Printf("warning: failed to parse state file: %v", err)
		return
	}
	c.experiments = st.Experiments
	log.Printf("loaded %d experiments from state", len(c.experiments))
}

func (c *Coordinator) saveState() {
	st := stateData{Experiments: c.experiments}
	data, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		log.Printf("error marshaling state: %v", err)
		return
	}
	tmp := c.stateFile + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		log.Printf("error writing state: %v", err)
		return
	}
	if err := os.Rename(tmp, c.stateFile); err != nil {
		log.Printf("error renaming state: %v", err)
	}
}

func (c *Coordinator) findExperiment(id string) *Experiment {
	for _, e := range c.experiments {
		if e.ID == id {
			return e
		}
	}
	return nil
}

// maxConsecutiveErrors is the threshold after which an experiment is paused.
const maxConsecutiveErrors = 3

// activeExperiment returns a random running experiment for load balancing.
// Skips experiments that have hit the consecutive error limit.
func (c *Coordinator) activeExperiment() *Experiment {
	var candidates []*Experiment
	for _, e := range c.experiments {
		if e.Result.Status == "" || e.Result.Status == "running" {
			if e.ConsecutiveErrors >= maxConsecutiveErrors {
				continue // paused due to errors
			}
			candidates = append(candidates, e)
		}
	}
	if len(candidates) == 0 {
		return nil
	}
	return candidates[time.Now().UnixNano()%int64(len(candidates))]
}

// Handler methods

func (c *Coordinator) handleListExperiments(w http.ResponseWriter, r *http.Request) {
	c.mu.Lock()
	defer c.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	exps := c.experiments
	if exps == nil {
		exps = []*Experiment{}
	}
	json.NewEncoder(w).Encode(exps)
}

func (c *Coordinator) handleCreateExperiment(w http.ResponseWriter, r *http.Request) {
	var e Experiment
	if err := json.NewDecoder(r.Body).Decode(&e); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	if e.ID == "" || e.Branch == "" {
		http.Error(w, "id and branch are required", http.StatusBadRequest)
		return
	}

	// Apply defaults
	if e.BaseBranch == "" {
		e.BaseBranch = "main"
	}
	if e.SPRT == (SPRTConfig{}) {
		e.SPRT = DefaultSPRT()
	}
	if e.TC == "" {
		e.TC = "10+0.1"
	}
	if e.HashMB == 0 {
		e.HashMB = 64
	}
	if e.Concurrency == 0 {
		e.Concurrency = 1
	}
	if e.BatchSize == 0 {
		e.BatchSize = 50
	}
	e.CreatedAt = time.Now()
	e.Result.Status = "running"

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.findExperiment(e.ID) != nil {
		http.Error(w, "experiment already exists: "+e.ID, http.StatusConflict)
		return
	}

	c.experiments = append(c.experiments, &e)
	c.saveState()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(e)
	log.Printf("created experiment: %s (branch=%s)", e.ID, e.Branch)
}

func (c *Coordinator) handleDeleteExperiment(w http.ResponseWriter, r *http.Request) {
	id := filepath.Base(r.URL.Path) // /api/experiments/{id}

	c.mu.Lock()
	defer c.mu.Unlock()

	for i, e := range c.experiments {
		if e.ID == id {
			c.experiments = append(c.experiments[:i], c.experiments[i+1:]...)
			// Remove pending claims for this experiment
			var kept []PendingClaim
			for _, pc := range c.pendingClaims {
				if pc.ExperimentID != id {
					kept = append(kept, pc)
				}
			}
			c.pendingClaims = kept
			c.saveState()
			log.Printf("deleted experiment: %s", id)
			w.WriteHeader(http.StatusNoContent)
			return
		}
	}
	http.Error(w, "not found", http.StatusNotFound)
}

func (c *Coordinator) handleClaimWork(w http.ResponseWriter, r *http.Request) {
	var req ClaimRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	e := c.activeExperiment()
	if e == nil {
		w.WriteHeader(http.StatusNoContent) // no work available
		return
	}

	claim := WorkClaim{
		ExperimentID: e.ID,
		Branch:       e.Branch,
		BaseBranch:   e.BaseBranch,
		TC:           e.TC,
		HashMB:       e.HashMB,
		Concurrency:  e.Concurrency,
		BatchSize:    e.BatchSize,
		OpeningStart: e.NextOpening,
		RepoURL:      c.repoURL,
		NNUEFile:     e.NNUEFile,
		BaseNNUEFile: e.BaseNNUEFile,
		Options:      e.Options,
	}

	e.NextOpening += e.BatchSize

	c.pendingClaims = append(c.pendingClaims, PendingClaim{
		ExperimentID: e.ID,
		WorkerID:     req.WorkerID,
		OpeningStart: claim.OpeningStart,
		BatchSize:    e.BatchSize,
		ClaimedAt:    time.Now(),
	})

	c.saveState()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(claim)
}

func (c *Coordinator) handleReportWork(w http.ResponseWriter, r *http.Request) {
	var rep BatchReport
	if err := json.NewDecoder(r.Body).Decode(&rep); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	e := c.findExperiment(rep.ExperimentID)
	if e == nil {
		http.Error(w, "experiment not found: "+rep.ExperimentID, http.StatusNotFound)
		return
	}

	// Remove corresponding pending claim
	for i, pc := range c.pendingClaims {
		if pc.ExperimentID == rep.ExperimentID && pc.WorkerID == rep.WorkerID {
			c.pendingClaims = append(c.pendingClaims[:i], c.pendingClaims[i+1:]...)
			break
		}
	}

	// Record batch
	e.Batches = append(e.Batches, BatchRecord{
		WorkerID:  rep.WorkerID,
		Wins:      rep.Wins,
		Draws:     rep.Draws,
		Losses:    rep.Losses,
		Error:     rep.Error,
		Timestamp: time.Now(),
	})

	// Update totals and error tracking
	if rep.Error != "" {
		e.ConsecutiveErrors++
		if e.ConsecutiveErrors >= maxConsecutiveErrors {
			e.Result.Status = "error"
			log.Printf("experiment %s paused: %d consecutive errors (last: %s)",
				e.ID, e.ConsecutiveErrors, rep.Error)
		}
	} else {
		e.ConsecutiveErrors = 0
		e.Result.Wins += rep.Wins
		e.Result.Draws += rep.Draws
		e.Result.Losses += rep.Losses
		UpdateSPRT(&e.Result, e.SPRT)
	}

	total := e.Result.Wins + e.Result.Draws + e.Result.Losses
	log.Printf("batch report: experiment=%s worker=%s +%d=%d -%d total=%d LLR=%.2f Elo=%.1f±%.1f [%s]",
		rep.ExperimentID, rep.WorkerID, rep.Wins, rep.Draws, rep.Losses,
		total, e.Result.LLR, e.Result.Elo, e.Result.EloErr, e.Result.Status)

	c.saveState()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(e.Result)
}

// RegisterRoutes sets up the HTTP handlers.
func (c *Coordinator) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/api/experiments", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			c.handleListExperiments(w, r)
		case http.MethodPost:
			c.handleCreateExperiment(w, r)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})

	// DELETE /api/experiments/{id}
	mux.HandleFunc("/api/experiments/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			c.handleDeleteExperiment(w, r)
		} else if r.Method == http.MethodGet {
			// Single experiment
			c.mu.Lock()
			id := filepath.Base(r.URL.Path)
			e := c.findExperiment(id)
			c.mu.Unlock()
			if e == nil {
				http.Error(w, "not found", http.StatusNotFound)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(e)
		} else {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/api/work/claim", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		c.handleClaimWork(w, r)
	})

	mux.HandleFunc("/api/work/report", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		c.handleReportWork(w, r)
	})
}

// StatusSummary returns a formatted summary of all experiments.
func (c *Coordinator) StatusSummary() string {
	c.mu.Lock()
	defer c.mu.Unlock()

	var running, finished []*Experiment
	for _, e := range c.experiments {
		switch e.Result.Status {
		case "running", "", "error":
			running = append(running, e)
		default:
			finished = append(finished, e)
		}
	}

	sort.Slice(finished, func(i, j int) bool {
		return finished[i].CreatedAt.After(finished[j].CreatedAt)
	})

	s := ""
	if len(running) > 0 {
		s += "Running:\n"
		for _, e := range running {
			total := e.Result.Wins + e.Result.Draws + e.Result.Losses
			s += fmt.Sprintf("  %-20s %d games  Elo %.1f±%.1f  LLR %.2f [%.2f, %.2f]\n",
				e.ID, total, e.Result.Elo, e.Result.EloErr, e.Result.LLR,
				e.SPRT.LowerBound(), e.SPRT.UpperBound())
		}
	}
	if len(finished) > 0 {
		s += "Finished:\n"
		for _, e := range finished {
			total := e.Result.Wins + e.Result.Draws + e.Result.Losses
			s += fmt.Sprintf("  %-20s %d games  Elo %.1f±%.1f  %s\n",
				e.ID, total, e.Result.Elo, e.Result.EloErr, e.Result.Status)
		}
	}
	if s == "" {
		s = "No experiments.\n"
	}
	return s
}
