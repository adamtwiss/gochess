package chess

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// NetTxtFilename reads net.txt (from the same directory as the executable, or CWD)
// and extracts the expected net filename from the URL.
func NetTxtFilename() (string, error) {
	// Try net.txt next to the binary first
	exePath, _ := os.Executable()
	exeDir := filepath.Dir(exePath)

	url, err := readNetTxt(filepath.Join(exeDir, "net.txt"))
	if err != nil {
		// Fall back to CWD
		url, err = readNetTxt("net.txt")
		if err != nil {
			return "", fmt.Errorf("net.txt not found (tried %s and CWD)", exeDir)
		}
	}

	// Extract filename from URL
	parts := strings.Split(strings.TrimSpace(url), "/")
	if len(parts) == 0 {
		return "", fmt.Errorf("net.txt contains empty URL")
	}
	filename := parts[len(parts)-1]
	if filename == "" {
		return "", fmt.Errorf("net.txt URL has no filename")
	}
	return filename, nil
}

// NetTxtURL reads net.txt and returns the URL.
func NetTxtURL() (string, error) {
	exePath, _ := os.Executable()
	exeDir := filepath.Dir(exePath)

	url, err := readNetTxt(filepath.Join(exeDir, "net.txt"))
	if err != nil {
		url, err = readNetTxt("net.txt")
		if err != nil {
			return "", fmt.Errorf("net.txt not found")
		}
	}
	return strings.TrimSpace(url), nil
}

func readNetTxt(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	url := strings.TrimSpace(string(data))
	if url == "" {
		return "", fmt.Errorf("net.txt is empty")
	}
	return url, nil
}

// LoadNNUEFromNetTxt resolves and loads the NNUE net specified by net.txt.
// It searches for the file in: (1) executable directory, (2) CWD.
// Returns the loaded net (v4 or v5), the path it was loaded from, and any error.
// If the net file is not found, returns a descriptive error.
func LoadNNUEFromNetTxt() (nnueNet *NNUENet, nnueNetV5 *NNUENetV5, loadedPath string, err error) {
	filename, err := NetTxtFilename()
	if err != nil {
		return nil, nil, "", fmt.Errorf("cannot determine net filename: %w", err)
	}

	exePath, _ := os.Executable()
	exeDir := filepath.Dir(exePath)

	// Search paths: executable directory, then CWD
	candidates := []string{
		filepath.Join(exeDir, filename),
		filename,
	}

	for _, path := range candidates {
		if _, statErr := os.Stat(path); statErr != nil {
			continue
		}

		// Detect version and load
		version, verr := DetectNNUEVersion(path)
		if verr != nil {
			return nil, nil, "", fmt.Errorf("error reading %s: %w", path, verr)
		}

		if version == 5 || version == 6 || version == 7 {
			netV5, lerr := LoadNNUEV5(path)
			if lerr != nil {
				return nil, nil, "", fmt.Errorf("error loading NNUE v5 from %s: %w", path, lerr)
			}
			return nil, netV5, path, nil
		}

		net, lerr := LoadNNUEAnyVersion(path)
		if lerr != nil {
			return nil, nil, "", fmt.Errorf("error loading NNUE from %s: %w", path, lerr)
		}
		return net, nil, path, nil
	}

	return nil, nil, "", fmt.Errorf("NNUE net '%s' not found. Run './chess fetch-net' to download it", filename)
}
