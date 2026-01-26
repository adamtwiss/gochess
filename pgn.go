package chess

import (
	"bufio"
	"io"
	"os"
	"regexp"
	"strings"
)

// PGNGame represents a single parsed PGN game.
type PGNGame struct {
	Tags  map[string]string // Tag pairs: "ECO" -> "A00", "Opening" -> "Polish", etc.
	Moves []string          // SAN move tokens: ["e4", "e5", "Nf3", ...]
}

var (
	tagRegexp    = regexp.MustCompile(`^\[(\w+)\s+"(.*)"\]$`)
	moveNumRegex = regexp.MustCompile(`^\d+\.+$`)
	resultRegex  = regexp.MustCompile(`^(1-0|0-1|1/2-1/2|\*)$`)
)

// ParsePGN parses all games from a PGN-formatted reader.
func ParsePGN(r io.Reader) ([]PGNGame, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 256*1024), 256*1024)

	var games []PGNGame
	var current *PGNGame
	inBraceComment := false

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// Handle multi-line brace comments
		if inBraceComment {
			if idx := strings.Index(line, "}"); idx >= 0 {
				inBraceComment = false
				line = strings.TrimSpace(line[idx+1:])
				if line == "" {
					continue
				}
			} else {
				continue
			}
		}

		// Check if entire line starts a brace comment (no tag/move context)
		if strings.HasPrefix(line, "{") {
			if idx := strings.Index(line, "}"); idx >= 0 {
				line = strings.TrimSpace(line[idx+1:])
				if line == "" {
					continue
				}
			} else {
				inBraceComment = true
				continue
			}
		}

		// Skip empty lines
		if line == "" {
			continue
		}

		// Skip semicolon comments
		if strings.HasPrefix(line, ";") {
			continue
		}

		// Tag line
		if m := tagRegexp.FindStringSubmatch(line); m != nil {
			if current == nil || len(current.Moves) > 0 {
				// Start a new game
				games = append(games, PGNGame{Tags: make(map[string]string)})
				current = &games[len(games)-1]
			}
			current.Tags[m[1]] = m[2]
			continue
		}

		// Move text line
		if current == nil {
			// Stray move text without tags — start a game anyway
			games = append(games, PGNGame{Tags: make(map[string]string)})
			current = &games[len(games)-1]
		}

		tokens := tokenizeMoveText(line)
		for _, tok := range tokens {
			if resultRegex.MatchString(tok) {
				// Game result — finish this game
				current = nil
				break
			}
			current.Moves = append(current.Moves, tok)
		}
	}

	return games, scanner.Err()
}

// ParsePGNFile parses all games from a PGN file.
func ParsePGNFile(filename string) ([]PGNGame, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ParsePGN(f)
}

// tokenizeMoveText splits a line of move text into SAN tokens, stripping
// move numbers, inline brace comments, NAGs ($N), and annotation glyphs.
func tokenizeMoveText(line string) []string {
	var tokens []string
	// Remove inline brace comments
	for {
		start := strings.Index(line, "{")
		if start < 0 {
			break
		}
		end := strings.Index(line[start:], "}")
		if end < 0 {
			// Unterminated brace comment — drop rest of line
			line = line[:start]
			break
		}
		line = line[:start] + line[start+end+1:]
	}

	for _, tok := range strings.Fields(line) {
		// Skip move numbers (e.g., "1.", "12...", "1...")
		if moveNumRegex.MatchString(tok) {
			continue
		}
		// Skip NAGs ($1, $23, etc.)
		if strings.HasPrefix(tok, "$") {
			continue
		}
		// Strip trailing annotation glyphs (!, ?, !!, ??, !?, ?!)
		tok = strings.TrimRight(tok, "!?")
		if tok == "" {
			continue
		}
		// Check for result tokens
		if resultRegex.MatchString(tok) {
			tokens = append(tokens, tok)
			continue
		}
		tokens = append(tokens, tok)
	}
	return tokens
}
