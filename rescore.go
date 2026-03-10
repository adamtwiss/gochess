package chess

import (
	"encoding/binary"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// RescoreConfig controls in-place rescoring of .bin training data.
type RescoreConfig struct {
	DataFile    string   // .bin file to rescore in-place
	Depth       int      // fixed search depth
	Concurrency int      // number of parallel workers
	HashMB      int      // shared TT size in MB
	NNUENet     *NNUENet // NNUE network (shared read-only)
	SyzygyPath  string   // path to Syzygy tablebases (empty = disabled)
}

// RescoreTrainingData rescores a .bin training data file in-place.
// Each 32-byte record's 2-byte score field (offset 28) is overwritten
// with the search score at the configured depth. The file is modified
// in-place so a crash leaves a partially-rescored but valid file;
// restarting simply rescores already-rescored positions (idempotent).
func RescoreTrainingData(cfg RescoreConfig, onProgress func(done, total int)) error {
	// Validate file
	stat, err := os.Stat(cfg.DataFile)
	if err != nil {
		return fmt.Errorf("stat %s: %w", cfg.DataFile, err)
	}
	fileSize := stat.Size()
	if fileSize%BinpackRecordSize != 0 {
		return fmt.Errorf("%s: file size %d is not a multiple of %d", cfg.DataFile, fileSize, BinpackRecordSize)
	}
	totalRecords := int(fileSize / BinpackRecordSize)
	if totalRecords == 0 {
		return fmt.Errorf("%s: empty file", cfg.DataFile)
	}

	concurrency := cfg.Concurrency
	if concurrency <= 0 {
		concurrency = 1
	}

	// Shared TT across all workers
	hashMB := cfg.HashMB
	if hashMB <= 0 {
		hashMB = 64
	}
	tt := NewTranspositionTable(hashMB)

	// Partition records into contiguous chunks (preserves TT locality)
	type chunk struct {
		start, count int
	}
	chunks := make([]chunk, concurrency)
	perWorker := totalRecords / concurrency
	remainder := totalRecords % concurrency
	offset := 0
	for i := 0; i < concurrency; i++ {
		n := perWorker
		if i < remainder {
			n++
		}
		chunks[i] = chunk{start: offset, count: n}
		offset += n
	}

	var progressDone int64
	var mu sync.Mutex // protects onProgress callback
	var wg sync.WaitGroup

	for w := 0; w < concurrency; w++ {
		wg.Add(1)
		go func(workerID int, ch chunk) {
			defer wg.Done()

			// Each worker opens the file independently for read-write
			f, err := os.OpenFile(cfg.DataFile, os.O_RDWR, 0)
			if err != nil {
				fmt.Fprintf(os.Stderr, "worker %d: open error: %v\n", workerID, err)
				return
			}
			defer f.Close()

			// Per-worker board and search info
			var b Board
			info := &SearchInfo{
				TT: tt,
			}

			// Set up NNUE if configured
			var nnueAcc *NNUEAccumulatorStack
			if cfg.NNUENet != nil {
				UseNNUE = true
				nnueAcc = NewNNUEAccumulatorStack(256)
			}

			var rec [BinpackRecordSize]byte
			var scoreBuf [2]byte

			for i := 0; i < ch.count; i++ {
				recIdx := ch.start + i
				fileOffset := int64(recIdx) * BinpackRecordSize

				// Read the record
				if _, err := f.ReadAt(rec[:], fileOffset); err != nil {
					fmt.Fprintf(os.Stderr, "worker %d: read error at record %d: %v\n", workerID, recIdx, err)
					continue
				}

				// Unpack position
				board, _, _, err := UnpackPosition(rec)
				if err != nil {
					fmt.Fprintf(os.Stderr, "worker %d: unpack error at record %d: %v\n", workerID, recIdx, err)
					continue
				}

				// Copy board state
				b = *board

				// Set up NNUE accumulator
				if cfg.NNUENet != nil {
					b.NNUENet = cfg.NNUENet
					b.NNUEAcc = nnueAcc
					cfg.NNUENet.RecomputeAccumulator(nnueAcc.Current(), &b)
				}

				// Search at fixed depth with safety timeout
				info.StartTime = time.Now()
				info.MaxTime = 30 * time.Second
				atomic.StoreInt64(&info.Deadline, time.Now().Add(30*time.Second).UnixNano())
				atomic.StoreInt32(&info.Stopped, 0)
				info.Nodes = 0

				_, result := b.SearchWithInfo(cfg.Depth, info)

				// Get score (side-to-move relative from search)
				score := result.Score

				// Convert to White-relative
				if b.SideToMove == Black {
					score = -score
				}

				// Clamp mate/TB scores to ±1000 (same as selfplay)
				if score > 20000 {
					score = 1000
				} else if score < -20000 {
					score = -1000
				}

				// Clamp to binpack range
				s := int16(score)
				if s > binpackMaxScore {
					s = binpackMaxScore
				}
				if s < binpackMinScore {
					s = binpackMinScore
				}

				// Write the 2-byte score at offset 28
				binary.LittleEndian.PutUint16(scoreBuf[:], uint16(s))
				if _, err := f.WriteAt(scoreBuf[:], fileOffset+28); err != nil {
					fmt.Fprintf(os.Stderr, "worker %d: write error at record %d: %v\n", workerID, recIdx, err)
					continue
				}

				// Progress reporting
				done := atomic.AddInt64(&progressDone, 1)
				if onProgress != nil && (done%1000 == 0 || int(done) == totalRecords) {
					mu.Lock()
					onProgress(int(done), totalRecords)
					mu.Unlock()
				}
			}
		}(w, chunks[w])
	}

	wg.Wait()
	return nil
}
