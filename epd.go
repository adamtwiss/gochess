package chess

import (
	"bufio"
	"fmt"
	"os"
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
}

// RunEPDTest runs search on an EPD position and checks if it finds a best move
func RunEPDTest(epd *EPDPosition, depth int, maxTime time.Duration, tt *TranspositionTable) (*EPDTestResult, error) {
	var b Board
	// EPD only has 4 fields, add defaults for halfmove and fullmove
	fullFEN := epd.FEN + " 0 1"
	if err := b.SetFEN(fullFEN); err != nil {
		return nil, fmt.Errorf("invalid FEN: %w", err)
	}

	start := time.Now()
	bestMove, info := b.SearchWithTT(depth, maxTime, tt)
	elapsed := time.Since(start)

	result := &EPDTestResult{
		Position:    epd,
		BestMove:    bestMove,
		BestMoveSAN: b.ToSAN(bestMove),
		SearchInfo:  info,
		TimeTaken:   elapsed,
	}

	// Check if the found move is in the best moves list
	for _, bm := range epd.BestMoves {
		expectedMove, err := b.ParseSAN(bm)
		if err != nil {
			continue
		}
		if bestMove == expectedMove {
			result.Passed = true
			break
		}
	}

	// Also check if the move avoids the "avoid moves"
	if result.Passed && len(epd.AvoidMoves) > 0 {
		for _, am := range epd.AvoidMoves {
			avoidMove, err := b.ParseSAN(am)
			if err != nil {
				continue
			}
			if bestMove == avoidMove {
				result.Passed = false
				break
			}
		}
	}

	return result, nil
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
