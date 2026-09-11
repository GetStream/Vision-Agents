package research

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

var codeExtensions = map[string]bool{
	".swift": true, ".ts": true, ".tsx": true, ".js": true, ".jsx": true, ".mjs": true, ".cjs": true,
	".go": true, ".py": true, ".pyi": true, ".kt": true, ".kts": true, ".java": true, ".dart": true,
	".cs": true, ".c": true, ".cpp": true, ".cc": true, ".cxx": true, ".h": true, ".hpp": true, ".hh": true,
	".m": true, ".mm": true, ".php": true, ".rb": true, ".proto": true,
}
var extensions = func() map[string]bool {
	result := map[string]bool{".json": true, ".md": true, ".mdx": true, ".yaml": true, ".yml": true, ".toml": true, ".xml": true, ".gradle": true, ".csproj": true, ".props": true, ".txt": true}
	for extension := range codeExtensions {
		result[extension] = true
	}
	return result
}()

const maxIndexBytes = 100_000_000
const maxIndexFiles = 20000

var words = regexp.MustCompile(`[A-Za-z][A-Za-z0-9_]{3,}`)

type SourceFile struct {
	RepositoryID string
	Path         string
	Text         string
}
type Index struct {
	Root  string
	Files []SourceFile
}

func ReadSource(root, repo, path string) (string, error) {
	if !identifier.MatchString(repo) || filepath.IsAbs(path) || strings.Contains(path, "\\") || path == "." || strings.HasPrefix(path, "../") || strings.Contains(path, "/../") || !extensions[filepath.Ext(path)] {
		return "", errors.New("invalid source path")
	}
	for _, part := range strings.Split(path, "/") {
		if strings.HasPrefix(part, ".") {
			return "", errors.New("hidden source path")
		}
	}
	base := filepath.Join(root, repo)
	name := filepath.Join(base, path)
	real, e := filepath.EvalSymlinks(name)
	if e != nil {
		return "", e
	}
	if real != name || !strings.HasPrefix(real, base+string(os.PathSeparator)) {
		return "", errors.New("source symlink")
	}
	info, e := os.Stat(name)
	if e != nil {
		return "", e
	}
	if !info.Mode().IsRegular() || info.Size() > 2_000_000 {
		return "", errors.New("source too large")
	}
	b, e := os.ReadFile(name)
	return string(b), e
}

// The excerpt index is a bounded accelerator, not the source access allowlist.
// Split its budget fairly so adding a large SDK cannot starve the other repositories.
func BuildIndex(root string, repos []Repository) (*Index, error) {
	index := &Index{Root: root}
	bytesPerRepo := maxIndexBytes / max(1, len(repos))
	filesPerRepo := max(1, maxIndexFiles/max(1, len(repos)))
	for _, repo := range repos {
		var candidates []string
		e := filepath.WalkDir(filepath.Join(root, repo.ID), func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				if strings.HasPrefix(entry.Name(), ".") || entry.Name() == "node_modules" {
					return filepath.SkipDir
				}
				return nil
			}
			if entry.Type()&os.ModeSymlink != 0 || !extensions[filepath.Ext(path)] {
				return nil
			}
			rel, err := filepath.Rel(filepath.Join(root, repo.ID), path)
			if err != nil {
				return err
			}
			candidates = append(candidates, rel)
			return nil
		})
		if e != nil {
			return nil, e
		}
		sort.SliceStable(candidates, func(i, j int) bool { return sourcePriority(candidates[i]) < sourcePriority(candidates[j]) })
		before, used := len(index.Files), 0
		for _, path := range candidates {
			if len(index.Files)-before >= filesPerRepo {
				break
			}
			text, err := ReadSource(root, repo.ID, path)
			if err != nil || len(text)+used > bytesPerRepo {
				continue
			}
			used += len(text)
			index.Files = append(index.Files, SourceFile{repo.ID, path, text})
		}
		if len(index.Files) == before {
			return nil, fmt.Errorf("repository %s has no indexable source", repo.ID)
		}
	}
	return index, nil
}
func sourcePriority(path string) int {
	if !codeExtensions[filepath.Ext(path)] {
		return 3
	}
	lower := "/" + strings.ToLower(filepath.ToSlash(path))
	for _, directory := range []string{"/test", "/example", "/sample", "/demo"} {
		if strings.Contains(lower, directory) {
			return 2
		}
	}
	return 0
}
func (idx *Index) Context(question string, repos []Repository) string {
	allowed := map[string]bool{}
	for _, r := range repos {
		allowed[r.ID] = true
	}
	type hit struct {
		score int
		text  string
	}
	var hits []hit
	stop := map[string]bool{"what": true, "does": true, "where": true, "which": true, "with": true, "from": true, "when": true, "this": true, "that": true, "defined": true, "return": true, "renders": true}
	var tokens []string
	for _, t := range words.FindAllString(question, -1) {
		if !stop[strings.ToLower(t)] {
			tokens = append(tokens, strings.ToLower(t))
		}
	}
	for _, f := range idx.Files {
		if !allowed[f.RepositoryID] {
			continue
		}
		lines := strings.Split(f.Text, "\n")
		type location struct{ line, score int }
		var best []location
		pathScore := 0
		for _, token := range tokens {
			if strings.Contains(strings.ToLower(f.Path), token) {
				pathScore += 2
			}
		}
		for n, line := range lines {
			score := pathScore
			matches := 0
			for _, token := range tokens {
				if strings.Contains(strings.ToLower(line), token) {
					score += 3
					matches++
				}
			}
			if matches == 0 {
				continue
			}
			best = append(best, location{n, score})
		}
		sort.SliceStable(best, func(i, j int) bool { return best[i].score > best[j].score })
		var chosen []int
		for _, b := range best {
			overlap := false
			for _, n := range chosen {
				if b.line >= n-8 && b.line <= n+8 {
					overlap = true
				}
			}
			if overlap {
				continue
			}
			chosen = append(chosen, b.line)
			start := max(0, b.line-4)
			end := min(len(lines), b.line+12)
			var excerpt strings.Builder
			fmt.Fprintf(&excerpt, "%s/%s\n", f.RepositoryID, f.Path)
			for i := start; i < end; i++ {
				fmt.Fprintf(&excerpt, "%d: %s\n", i+1, lines[i])
			}
			hits = append(hits, hit{b.score, excerpt.String()})
			if len(chosen) == 2 {
				break
			}
		}
		if len(hits) >= 4000 {
			break
		}
	}
	sort.SliceStable(hits, func(i, j int) bool { return hits[i].score > hits[j].score })
	var out strings.Builder
	for _, h := range hits {
		if out.Len()+len(h.text) > 12000 {
			continue
		}
		out.WriteString(h.text)
		out.WriteByte('\n')
		if out.Len() > 10000 {
			break
		}
	}
	return out.String()
}
func Verify(text string, root string, repos []Repository) (Result, error) {
	var result Result
	text = strings.TrimSpace(text)
	text = strings.TrimPrefix(text, "```json")
	text = strings.TrimPrefix(text, "```")
	text = strings.TrimSuffix(text, "```")
	if len(text) > 64000 || json.Unmarshal([]byte(text), &result) != nil || (result.Status != "answered" && result.Status != "insufficient_evidence") || result.Answer == "" || len(result.Citations) > 8 || (result.Status == "answered" && len(result.Citations) == 0) {
		return Result{}, errors.New("cursor_invalid_output")
	}
	allowed := map[string]Repository{}
	for _, r := range repos {
		allowed[r.ID] = r
	}
	for i := range result.Citations {
		c := &result.Citations[i]
		r, ok := allowed[c.RepositoryID]
		if !ok || !sha.MatchString(r.Revision) || len(c.Quote) < 8 || len(c.Quote) > 1500 {
			return Result{}, errors.New("citation_mismatch")
		}
		source, e := ReadSource(root, r.ID, c.Path)
		if e != nil {
			return Result{}, errors.New("citation_mismatch")
		}
		// Derive authoritative lines from a unique exact quote, not model-counted lines.
		offset := strings.Index(source, c.Quote)
		if offset < 0 || strings.Contains(source[offset+1:], c.Quote) {
			return Result{}, errors.New("citation_mismatch")
		}
		c.StartLine = strings.Count(source[:offset], "\n") + 1
		c.EndLine = c.StartLine + strings.Count(c.Quote, "\n")
		c.Revision = r.Revision
		parts := strings.Split(c.Path, "/")
		for n := range parts {
			parts[n] = url.PathEscape(parts[n])
		}
		c.URL = fmt.Sprintf("%s/blob/%s/%s#L%d-L%d", strings.TrimSuffix(r.URL, ".git"), r.Revision, strings.Join(parts, "/"), c.StartLine, c.EndLine)
	}
	result.Code = ""
	result.ElapsedMS = 0
	return result, nil
}
