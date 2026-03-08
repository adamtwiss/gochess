package chess

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"time"
)

// BenchmarkSolvedBonus is the base score awarded for solving a position,
// on top of the log2(timeLimit/solveTime) speed bonus.
const BenchmarkSolvedBonus = 1.0

// BenchmarkScore computes a continuous quality score for a single position.
// If solved: 1.0 + log2(timeLimitMs / solveTimeMs).
// If unsolved: 0.
// solveTimeMs is floored to 1 to avoid log2(inf).
func BenchmarkScore(timeLimitMs, solveTimeMs float64) float64 {
	if solveTimeMs <= 0 {
		return 0
	}
	if solveTimeMs < 1 {
		solveTimeMs = 1
	}
	return BenchmarkSolvedBonus + math.Log2(timeLimitMs/solveTimeMs)
}

// BenchmarkSuiteConfig defines an EPD suite to include in the benchmark.
type BenchmarkSuiteConfig struct {
	Name     string
	Filename string
}

// DefaultBenchmarkSuites returns the standard set of benchmark suites.
func DefaultBenchmarkSuites() []BenchmarkSuiteConfig {
	return []BenchmarkSuiteConfig{
		{Name: "WAC", Filename: "testdata/wac.epd"},
		{Name: "ECM", Filename: "testdata/ecm.epd"},
		{Name: "SBD", Filename: "testdata/sbd.epd"},
		{Name: "Arasan", Filename: "testdata/arasan.epd"},
	}
}

// BenchmarkPositionResult holds the result of benchmarking a single position.
type BenchmarkPositionResult struct {
	ID          string  `json:"id"`
	FEN         string  `json:"fen"`
	Expected    string  `json:"expected"`
	Found       string  `json:"found"`
	Solved      bool    `json:"solved"`
	Score       float64 `json:"score"`
	SolveTimeMs float64 `json:"solve_time_ms"`
	TotalTimeMs float64 `json:"total_time_ms"`
	Nodes       uint64  `json:"nodes"`
	Depth       int     `json:"depth"`
}

// BenchmarkSuiteResult holds the aggregate result for one EPD suite.
type BenchmarkSuiteResult struct {
	Name       string                    `json:"name"`
	Solved     int                       `json:"solved"`
	Total      int                       `json:"total"`
	Score      float64                   `json:"score"`
	TotalNodes uint64                    `json:"total_nodes"`
	DurationMs float64                   `json:"duration_ms"`
	Positions  []BenchmarkPositionResult `json:"positions"`
}

// BenchmarkResult holds the top-level benchmark results across all suites.
type BenchmarkResult struct {
	Timestamp   string                 `json:"timestamp"`
	TimeLimitMs int                    `json:"time_limit_ms"`
	HashMB      int                    `json:"hash_mb"`
	Threads     int                    `json:"threads"`
	Suites      []BenchmarkSuiteResult `json:"suites"`
}

// TotalSolved returns the total number of solved positions across all suites.
func (r *BenchmarkResult) TotalSolved() int {
	n := 0
	for _, s := range r.Suites {
		n += s.Solved
	}
	return n
}

// TotalPositions returns the total number of positions across all suites.
func (r *BenchmarkResult) TotalPositions() int {
	n := 0
	for _, s := range r.Suites {
		n += s.Total
	}
	return n
}

// TotalScore returns the sum of scores across all suites.
func (r *BenchmarkResult) TotalScore() float64 {
	sum := 0.0
	for _, s := range r.Suites {
		sum += s.Score
	}
	return sum
}

// TotalNodes returns the sum of nodes across all suites.
func (r *BenchmarkResult) TotalNodes() uint64 {
	n := uint64(0)
	for _, s := range r.Suites {
		n += s.TotalNodes
	}
	return n
}

// TotalDuration returns the sum of durations across all suites in milliseconds.
func (r *BenchmarkResult) TotalDuration() float64 {
	sum := 0.0
	for _, s := range r.Suites {
		sum += s.DurationMs
	}
	return sum
}

// BenchmarkProgress is passed to the RunBenchmark callback after each position.
type BenchmarkProgress struct {
	Suite       string
	PositionNum int
	TotalInSuite int
	ID          string
	Solved      bool
	Score       float64
}

// RunBenchmark runs the benchmark across all configured suites.
func RunBenchmark(suites []BenchmarkSuiteConfig, timeLimitMs, hashMB, depth, threads int, onPosition func(BenchmarkProgress)) (*BenchmarkResult, error) {
	result := &BenchmarkResult{
		Timestamp:   time.Now().UTC().Format(time.RFC3339),
		TimeLimitMs: timeLimitMs,
		HashMB:      hashMB,
		Threads:     threads,
	}

	maxTime := time.Duration(timeLimitMs) * time.Millisecond
	timeLimitF := float64(timeLimitMs)

	for _, suite := range suites {
		positions, err := LoadEPDFile(suite.Filename)
		if err != nil {
			return nil, fmt.Errorf("suite %s: %w", suite.Name, err)
		}

		tt := NewTranspositionTable(hashMB)
		sr := BenchmarkSuiteResult{
			Name:      suite.Name,
			Total:     len(positions),
			Positions: make([]BenchmarkPositionResult, 0, len(positions)),
		}

		suiteStart := time.Now()

		for i, pos := range positions {
			tt.Clear()

			epdResult, err := RunEPDTestWithInfo(pos, depth, maxTime, tt, nil, threads)
			if err != nil {
				return nil, fmt.Errorf("suite %s, position %d: %w", suite.Name, i+1, err)
			}

			id := pos.ID
			if id == "" {
				id = fmt.Sprintf("#%d", i+1)
			}

			pr := BenchmarkPositionResult{
				ID:          id,
				FEN:         pos.FEN,
				Expected:    strings.Join(pos.BestMoves, "/"),
				Found:       epdResult.BestMoveSAN,
				Solved:      epdResult.Passed,
				TotalTimeMs: float64(epdResult.TimeTaken.Milliseconds()),
				Nodes:       epdResult.SearchInfo.Nodes,
				Depth:       epdResult.SearchInfo.Depth,
			}

			if epdResult.Passed && epdResult.SolveTime > 0 {
				solveMs := float64(epdResult.SolveTime.Milliseconds())
				pr.SolveTimeMs = solveMs
				pr.Score = BenchmarkScore(timeLimitF, solveMs)
			}

			if pr.Solved {
				sr.Solved++
				sr.Score += pr.Score
			}
			sr.TotalNodes += pr.Nodes
			sr.Positions = append(sr.Positions, pr)

			if onPosition != nil {
				onPosition(BenchmarkProgress{
					Suite:        suite.Name,
					PositionNum:  i + 1,
					TotalInSuite: len(positions),
					ID:           id,
					Solved:       pr.Solved,
					Score:        pr.Score,
				})
			}
		}

		sr.DurationMs = float64(time.Since(suiteStart).Milliseconds())
		result.Suites = append(result.Suites, sr)
	}

	return result, nil
}

// SaveBenchmarkResult writes benchmark results to a JSON file.
func SaveBenchmarkResult(filename string, result *BenchmarkResult) error {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

// LoadBenchmarkResult reads benchmark results from a JSON file.
func LoadBenchmarkResult(filename string) (*BenchmarkResult, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	var result BenchmarkResult
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

