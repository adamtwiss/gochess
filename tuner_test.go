package chess

import (
	"bytes"
	"math"
	"testing"
)

func TestTbinV3RoundTrip(t *testing.T) {
	// Initialize tuner to get parameter catalog
	tuner := &Tuner{}
	tuner.initTunerParams()
	pairMap := tuner.BuildPairMap()

	if len(pairMap) == 0 {
		t.Fatal("BuildPairMap returned empty map")
	}
	t.Logf("Pair map: %d MG→EG pairs out of %d total params", len(pairMap), tuner.NumParams())

	// Verify pair map is self-consistent: no duplicate EG targets
	egSeen := make(map[uint16]uint16)
	for mg, eg := range pairMap {
		if prev, ok := egSeen[eg]; ok {
			t.Errorf("EG index %d mapped from both MG %d and MG %d", eg, prev, mg)
		}
		egSeen[eg] = mg
	}

	// Create test traces with various patterns
	tests := []struct {
		name  string
		trace TunerTrace
	}{
		{
			name: "all paired",
			trace: TunerTrace{
				Phase: 12, Result: 0.5, Score: 42, WScale: 128, BScale: 128, HalfmoveClock: 5,
				MG: []SparseEntry{{Index: uint16(idxPSTMG), Coeff: 1}, {Index: uint16(idxPSTMG + 1), Coeff: -1}},
				EG: []SparseEntry{{Index: uint16(idxPSTEG), Coeff: 1}, {Index: uint16(idxPSTEG + 1), Coeff: -1}},
			},
		},
		{
			name: "mixed paired and unpaired",
			trace: TunerTrace{
				Phase: 20, Result: 1.0, Score: -100, WScale: 100, BScale: 64, HalfmoveClock: 0,
				MG: []SparseEntry{
					{Index: uint16(idxMaterialMG), Coeff: 2},     // paired (mat MG[0] → EG[0])
					{Index: uint16(idxKingSafetyTbl), Coeff: 1},  // MG-only (king safety is MG-only)
				},
				EG: []SparseEntry{
					{Index: uint16(idxMaterialEG), Coeff: 2},     // paired with material MG
					{Index: uint16(idxEndgameKing), Coeff: 3},    // EG-only
				},
			},
		},
		{
			name: "empty trace",
			trace: TunerTrace{
				Phase: 0, Result: 0.0, Score: 0, WScale: 128, BScale: 128,
				MG: nil, EG: nil,
			},
		},
		{
			name: "different coefficients break pairing",
			trace: TunerTrace{
				Phase: 10, Result: 0.5, Score: 50, WScale: 128, BScale: 128,
				MG: []SparseEntry{{Index: uint16(idxPSTMG + 10), Coeff: 1}},
				EG: []SparseEntry{{Index: uint16(idxPSTEG + 10), Coeff: 2}}, // different coeff
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Encode
			var buf bytes.Buffer
			n, err := writeTraceRecord(&buf, &tt.trace, pairMap)
			if err != nil {
				t.Fatalf("writeTraceRecord: %v", err)
			}
			if n != buf.Len() {
				t.Fatalf("writeTraceRecord returned %d bytes but buffer has %d", n, buf.Len())
			}

			// Decode
			data := buf.Bytes()
			decoded, _, _, endOffset := decodeTraceRecord(data, 0, nil, nil, pairMap)
			if endOffset != len(data) {
				t.Errorf("decodeTraceRecord consumed %d bytes, expected %d", endOffset, len(data))
			}

			// Verify scalar fields
			if decoded.Phase != tt.trace.Phase {
				t.Errorf("Phase: got %d, want %d", decoded.Phase, tt.trace.Phase)
			}
			if decoded.Result != tt.trace.Result {
				t.Errorf("Result: got %f, want %f", decoded.Result, tt.trace.Result)
			}
			if decoded.Score != tt.trace.Score {
				t.Errorf("Score: got %d, want %d", decoded.Score, tt.trace.Score)
			}
			if decoded.WScale != tt.trace.WScale {
				t.Errorf("WScale: got %d, want %d", decoded.WScale, tt.trace.WScale)
			}
			if decoded.BScale != tt.trace.BScale {
				t.Errorf("BScale: got %d, want %d", decoded.BScale, tt.trace.BScale)
			}
			if decoded.HalfmoveClock != tt.trace.HalfmoveClock {
				t.Errorf("HalfmoveClock: got %d, want %d", decoded.HalfmoveClock, tt.trace.HalfmoveClock)
			}

			// Verify MG entries (order may differ due to paired-first layout)
			if len(decoded.MG) != len(tt.trace.MG) {
				t.Errorf("MG count: got %d, want %d", len(decoded.MG), len(tt.trace.MG))
			} else {
				origMG := make(map[uint16]int16)
				for _, e := range tt.trace.MG {
					origMG[e.Index] = e.Coeff
				}
				for _, e := range decoded.MG {
					if origMG[e.Index] != e.Coeff {
						t.Errorf("MG[%d]: got coeff %d, want %d", e.Index, e.Coeff, origMG[e.Index])
					}
					delete(origMG, e.Index)
				}
				if len(origMG) > 0 {
					t.Errorf("MG missing entries: %v", origMG)
				}
			}

			// Verify EG entries
			if len(decoded.EG) != len(tt.trace.EG) {
				t.Errorf("EG count: got %d, want %d", len(decoded.EG), len(tt.trace.EG))
			} else {
				origEG := make(map[uint16]int16)
				for _, e := range tt.trace.EG {
					origEG[e.Index] = e.Coeff
				}
				for _, e := range decoded.EG {
					if origEG[e.Index] != e.Coeff {
						t.Errorf("EG[%d]: got coeff %d, want %d", e.Index, e.Coeff, origEG[e.Index])
					}
					delete(origEG, e.Index)
				}
				if len(origEG) > 0 {
					t.Errorf("EG missing entries: %v", origEG)
				}
			}

			// For paired case, verify size is smaller than v2 would be
			v2Size := 11 + 4*len(tt.trace.MG) + 4*len(tt.trace.EG)
			t.Logf("v2 would be %d bytes, v3 is %d bytes (%.1f%% of v2)",
				v2Size, n, float64(n)/float64(v2Size)*100)
		})
	}
}

func TestTbinV3CompressionRatio(t *testing.T) {
	tuner := &Tuner{}
	tuner.initTunerParams()
	pairMap := tuner.BuildPairMap()

	fens := []string{
		"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
		"r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
		"r1bqkb1r/pppppppp/2n2n2/8/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
		"r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 b - - 5 8",
		"8/5ppk/4p2p/3pP3/R2P1PP1/4K2P/8/2r5 w - - 0 40",
	}

	params := tuner.Values // use current parameter values for score check

	var totalV2, totalV3 int
	for _, fen := range fens {
		b := &Board{}
		b.SetFEN(fen)
		trace := tuner.computeTrace(b)

		v2Size := 11 + 4*len(trace.MG) + 4*len(trace.EG)
		origScore := scoreFromTrace(&trace, params)

		var buf bytes.Buffer
		v3Size, err := writeTraceRecord(&buf, &trace, pairMap)
		if err != nil {
			t.Fatal(err)
		}

		// Round-trip verify: decoded trace must produce same score
		decoded, _, _, _ := decodeTraceRecord(buf.Bytes(), 0, nil, nil, pairMap)
		decodedScore := scoreFromTrace(&decoded, params)

		if math.Abs(origScore-decodedScore) > 0.001 {
			t.Errorf("Score mismatch for %s: orig=%.4f decoded=%.4f", fen, origScore, decodedScore)
		}

		totalV2 += v2Size
		totalV3 += v3Size
		t.Logf("%s: MG=%d→%d EG=%d→%d → v2=%d v3=%d (%.1f%%)",
			fen[:20], len(trace.MG), len(decoded.MG), len(trace.EG), len(decoded.EG),
			v2Size, v3Size, float64(v3Size)/float64(v2Size)*100)
	}

	ratio := float64(totalV3) / float64(totalV2) * 100
	t.Logf("Overall: v2=%d bytes, v3=%d bytes (%.1f%% of v2, %.1f%% reduction)",
		totalV2, totalV3, ratio, 100-ratio)

	if ratio > 70 {
		t.Errorf("Expected >30%% reduction, got only %.1f%% reduction", 100-ratio)
	}
}

func TestBuildPairMapCoverage(t *testing.T) {
	tuner := &Tuner{}
	tuner.initTunerParams()
	pairMap := tuner.BuildPairMap()

	// Check known pairs exist
	knownPairs := [][2]int{
		{idxMaterialMG, idxMaterialEG},         // first material pair
		{idxPSTMG, idxPSTEG},                   // first PST pair
		{idxMobilityStart, idxMobilityStart + 1}, // first mobility pair (interleaved)
		{idxBonusStart, idxBonusStart + 1},      // BishopPairMG → BishopPairEG
	}

	for _, pair := range knownPairs {
		mgIdx := uint16(pair[0])
		egIdx := uint16(pair[1])
		if got, ok := pairMap[mgIdx]; !ok {
			t.Errorf("Expected pair map entry for MG index %d, not found", mgIdx)
		} else if got != egIdx {
			t.Errorf("Pair map[%d] = %d, want %d", mgIdx, got, egIdx)
		}
	}

	// Check MG-only params are NOT in the map
	mgOnlyParams := []int{idxKingSafetyTbl, idxPawnShield, idxSameSideStorm}
	for _, idx := range mgOnlyParams {
		if _, ok := pairMap[uint16(idx)]; ok {
			t.Errorf("MG-only index %d should not be in pair map", idx)
		}
	}

	t.Logf("Total params: %d, paired: %d (%.1f%%)",
		tuner.NumParams(), len(pairMap), float64(len(pairMap)*2)/float64(tuner.NumParams())*100)
}
