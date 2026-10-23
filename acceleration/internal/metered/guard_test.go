package metered

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"maps"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// module is the import path every package of the tree is under.
const module = "github.com/GetStream/Vision-Agents/acceleration"

// facing is a type a vendor implements, which only the layer that meters it may hold.
type facing struct {
	// pkg is the package that declares it, relative to the module.
	pkg  string
	name string
}

// vendorFacing is every vendor-facing type, by name rather than found by reflection, so a
// component that adds one is a line here a reviewer sees.
var vendorFacing = []facing{
	{"internal/llm", "LLM"},
	{"internal/stt", "STT"},
	{"internal/tts", "TTS"},
	{"internal/sts", "STS"},
	{"internal/search", "Provider"},
	{"internal/imagegen", "Provider"},
	{"internal/detection", "Provider"},
	{"internal/decisionmodel", "Provider"},
	{"internal/tts/voices", "Cloner"},
	{"internal/phone", "Provider"},
	{"internal/channels", "Provider"},
	{"internal/dlc", "Registrar"},
}

// holders are, beside a type's own package and the vendor, suite and mock packages under it,
// the packages that may name a vendor-facing type: the routers and this package, which meter
// the work, the services that already meter theirs, the test audio, and the commands that
// build the process.
var holders = []string{
	"internal/metered",
	"internal/tts/voices",
	"internal/phone",
	"internal/channels",
	"internal/dlc",
	"internal/testaudio",
	"cmd",
}

// modalities are the packages whose vendors are packages under them, and the suite every one
// of those vendors runs.
var modalities = map[string]string{
	"internal/llm":           "internal/llm/llmsuite",
	"internal/stt":           "internal/stt/sttsuite",
	"internal/tts":           "internal/tts/ttssuite",
	"internal/sts":           "internal/sts/stssuite",
	"internal/search":        "internal/search/searchsuite",
	"internal/imagegen":      "internal/imagegen/imagegensuite",
	"internal/detection":     "internal/detection/detectionsuite",
	"internal/decisionmodel": "internal/decisionmodel/decisionmodelsuite",
}

// vendorRoots are the packages whose subpackages are vendors: the model modalities, and the
// phone and 10DLC vendors.
var vendorRoots = append(slices.Sorted(func(yield func(string) bool) {
	for modality := range modalities {
		if !yield(modality) {
			return
		}
	}
}), "internal/phone", "internal/dlc")

// importers are, beside the vendors themselves and the routers, which build theirs through
// their registries and meter what they build, the packages that may import a vendor: the
// phone vendors' registry, this package, the voice library, which meters its own cloning,
// and the commands.
var importers = []string{
	"internal/phone/vendors",
	"internal/metered",
	"internal/tts/voices",
	"cmd",
}

// importsAllowed are the files outside those that import a vendor, and why. The list may only
// shrink.
var importsAllowed = map[string]string{
	"internal/api/telnyxhook.go": "renders the call-control answer Telnyx's webhook asks for, and calls nothing",
}

// notVendors are packages under a vendor root that are not a vendor: what every vendor of
// the modality shares, and what tells it apart from the rest.
var notVendors = map[string]string{
	"internal/llm/openaicompat":        "the Chat Completions client every compatible vendor is built on",
	"internal/llm/openweights":         "what the hosts of open-weight models have in common",
	"internal/decisionmodel/systemone": "the System One protocol every decision model vendor answers on",
	"internal/tts/voices":              "the voice library and cloning service, a holder of its own",
	"internal/phone/sipbridge":         "the SIP bridge every phone vendor's calls go through",
	"internal/phone/siptrunk":          "the customer's own SIP trunk, which is no vendor of Stream's",
	"internal/phone/vendors":           "the phone vendors' registry",
}

// withoutSuite are the vendors that run no suite, and why. The list may only shrink.
var withoutSuite = map[string]string{
	"internal/llm/custom": "the customer's own endpoint, which no suite can reach",
	"internal/llm/moondream": "it takes no tools and is there to read pictures, so its own integration test " +
		"asks it what is in one and checks it reports its tokens",
	"internal/llm/cerebras": "no model of Cerebras's is in the router's config since gemma-4-31b was archived, " +
		"and the account answered payment_required (#1149): there is nothing to run the suite on",
}

// source is one non-test Go file of the tree, parsed.
type source struct {
	// dir is the file's package, relative to the module.
	dir  string
	name string
	file *ast.File
}

// tree parses every Go file under root's internal and cmd, test files apart, and the test
// files on their own.
func tree(t *testing.T, root string) (sources, tests []source) {
	t.Helper()
	files := token.NewFileSet()
	for _, top := range []string{"internal", "cmd"} {
		err := filepath.WalkDir(filepath.Join(root, top), func(name string, entry fs.DirEntry, err error) error {
			if err != nil {
				if os.IsNotExist(err) {
					return filepath.SkipDir
				}
				return err
			}
			if entry.IsDir() {
				if entry.Name() == "testdata" {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(name, ".go") {
				return nil
			}
			parsed, err := parser.ParseFile(files, name, nil, parser.SkipObjectResolution)
			if err != nil {
				return err
			}
			relative, err := filepath.Rel(root, filepath.Dir(name))
			if err != nil {
				return err
			}
			parsedSource := source{dir: filepath.ToSlash(relative), name: filepath.Base(name), file: parsed}
			if strings.HasSuffix(name, "_test.go") {
				tests = append(tests, parsedSource)
			} else {
				sources = append(sources, parsedSource)
			}
			return nil
		})
		require.NoError(t, err)
	}
	return sources, tests
}

// under reports whether dir is pkg or a package inside it.
func under(dir, pkg string) bool { return dir == pkg || strings.HasPrefix(dir, pkg+"/") }

// router reports whether dir is one of the routers, internal/<modality>router, or inside one.
func router(dir string) bool {
	parts := strings.Split(dir, "/")
	return len(parts) >= 2 && parts[0] == "internal" && strings.HasSuffix(parts[1], "router")
}

// imports is what each name a file refers to its imports by stands for, relative to the
// module, for the imports inside it.
func imports(file *ast.File) map[string]string {
	named := map[string]string{}
	for _, spec := range file.Imports {
		imported, err := strconv.Unquote(spec.Path.Value)
		if err != nil || !strings.HasPrefix(imported, module+"/") {
			continue
		}
		relative := strings.TrimPrefix(imported, module+"/")
		name := path.Base(relative)
		if spec.Name != nil {
			name = spec.Name.Name
		}
		named[name] = relative
	}
	return named
}

// vendorHolders is every file that names a vendor-facing type where it may not, as the file
// and the type.
func vendorHolders(sources []source) []string {
	var found []string
	for _, src := range sources {
		named := imports(src.file)
		ast.Inspect(src.file, func(node ast.Node) bool {
			selector, ok := node.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			qualifier, ok := selector.X.(*ast.Ident)
			if !ok {
				return true
			}
			pkg, ok := named[qualifier.Name]
			if !ok {
				return true
			}
			for _, vendor := range vendorFacing {
				if vendor.pkg == pkg && vendor.name == selector.Sel.Name && !mayHold(src.dir, vendor) {
					found = append(found, src.dir+"/"+src.name+" names "+path.Base(pkg)+"."+vendor.name)
				}
			}
			return true
		})
	}
	slices.Sort(found)
	return slices.Compact(found)
}

func mayHold(dir string, vendor facing) bool {
	if under(dir, vendor.pkg) || router(dir) {
		return true
	}
	return slices.ContainsFunc(holders, func(holder string) bool { return under(dir, holder) })
}

// vendorOf is the vendor package a package is, or is inside, and false for one that is none.
func vendorOf(dir string) (string, bool) {
	for _, root := range vendorRoots {
		if !strings.HasPrefix(dir, root+"/") {
			continue
		}
		vendor := root + "/" + strings.SplitN(strings.TrimPrefix(dir, root+"/"), "/", 2)[0]
		name := path.Base(vendor)
		if _, shared := notVendors[vendor]; shared || name == "mock" || strings.HasSuffix(name, "suite") || strings.HasSuffix(name, "test") {
			return "", false
		}
		return vendor, true
	}
	return "", false
}

// vendorImporters is every file that imports a vendor's package where it may not, as the
// file and the vendor.
func vendorImporters(sources []source) []string {
	var found []string
	for _, src := range sources {
		if _, allowed := importsAllowed[src.dir+"/"+src.name]; allowed {
			continue
		}
		if _, vendor := vendorOf(src.dir); vendor || router(src.dir) ||
			slices.ContainsFunc(importers, func(importer string) bool { return under(src.dir, importer) }) {
			continue
		}
		for _, imported := range imports(src.file) {
			if vendor, ok := vendorOf(imported); ok {
				found = append(found, src.dir+"/"+src.name+" imports "+vendor)
			}
		}
	}
	slices.Sort(found)
	return slices.Compact(found)
}

// vendorsWithoutSuite is every model vendor none of whose tests names its modality's suite,
// other than those withoutSuite gives a reason for.
func vendorsWithoutSuite(sources, tests []source) []string {
	vendors := map[string]bool{}
	for _, src := range sources {
		if vendor, ok := vendorOf(src.dir); ok && src.dir == vendor {
			if _, model := modalities[path.Dir(vendor)]; model {
				vendors[vendor] = false
			}
		}
	}
	for _, test := range tests {
		vendor, ok := vendorOf(test.dir)
		if !ok || test.dir != vendor {
			continue
		}
		suite := modalities[path.Dir(vendor)]
		for _, imported := range imports(test.file) {
			if imported == suite {
				vendors[vendor] = true
			}
		}
	}
	var found []string
	for vendor, runs := range vendors {
		if _, excused := withoutSuite[vendor]; !runs && !excused {
			found = append(found, vendor)
		}
	}
	slices.Sort(found)
	return found
}

// root is the module's directory.
func root(t *testing.T) string {
	t.Helper()
	dir, err := filepath.Abs("../..")
	require.NoError(t, err)
	_, err = os.Stat(filepath.Join(dir, "go.mod"))
	require.NoError(t, err, "the guard runs from internal/metered")
	return dir
}

func TestOnlyTheMeteringLayerHoldsAVendor(t *testing.T) {
	sources, _ := tree(t, root(t))

	require.Empty(t, vendorHolders(sources),
		"only the routers, internal/metered and the services that meter their own work may hold a vendor: "+
			"anything else calls it without its usage being recorded")
}

func TestOnlyRegistriesImportAVendor(t *testing.T) {
	sources, _ := tree(t, root(t))

	require.Empty(t, vendorImporters(sources),
		"a vendor is built by a registry, so the router that meters it is the one that calls it")
}

func TestEveryVendorRunsItsSuite(t *testing.T) {
	sources, tests := tree(t, root(t))

	require.Empty(t, vendorsWithoutSuite(sources, tests),
		"a vendor's suite is what checks it reports what it used; one that cannot run it is listed in withoutSuite with why")
}

func TestTheGuardCatchesWhatBreaksTheRule(t *testing.T) {
	sources, tests := tree(t, filepath.Join("testdata", "tree"))

	require.Equal(t, []string{"internal/api/voice.go names stt.STT"}, vendorHolders(sources))
	require.Equal(t, []string{"internal/agent/model.go imports internal/llm/openai"}, vendorImporters(sources))
	require.Equal(t, []string{"internal/llm/unsuited"}, vendorsWithoutSuite(sources, tests))
}

func TestTheExceptionsAreStillNeeded(t *testing.T) {
	sources, tests := tree(t, root(t))

	for file, why := range importsAllowed {
		index := slices.IndexFunc(sources, func(src source) bool { return src.dir+"/"+src.name == file })
		require.NotEqual(t, -1, index, "%s is gone, so its exception (%s) is too", file, why)
		imported := slices.ContainsFunc(slices.Collect(maps.Values(imports(sources[index].file))), func(pkg string) bool {
			_, vendor := vendorOf(pkg)
			return vendor
		})
		require.True(t, imported, "%s imports no vendor now, so it no longer needs its exception (%s)", file, why)
	}

	for vendor, why := range withoutSuite {
		for _, test := range tests {
			if test.dir != vendor {
				continue
			}
			for _, imported := range imports(test.file) {
				require.NotEqual(t, modalities[path.Dir(vendor)], imported,
					"%s runs its suite now, so it no longer needs its exception (%s)", vendor, why)
			}
		}
	}
}
