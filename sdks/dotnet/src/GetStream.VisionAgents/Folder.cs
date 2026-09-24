using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization;
using YamlDotNet.Core;
using YamlDotNet.RepresentationModel;
using YamlDotNet.Serialization;

namespace GetStream.VisionAgents;

/// <summary>
/// A kind of work worth handing to the slower model.
/// </summary>
/// <remarks>
/// There is nothing behind a skill but a better model and more time. What it declares is
/// the description the fast model chooses by, and the instructions the slow one answers
/// under.
/// </remarks>
public sealed record Skill
{
    /// <summary>How the fast model asks for it.</summary>
    public required string Name { get; init; }

    /// <summary>The one line the fast model sees.</summary>
    public required string Description { get; init; }

    /// <summary>The full prompt, which only the subagent sees.</summary>
    public required string Instructions { get; init; }

    /// <summary>Whether the work is handed recent video frames.</summary>
    public bool CaptureVideo { get; init; }

    /// <summary>How long the work may run before it is abandoned. Null leaves the backend's default.</summary>
    public TimeSpan? Deadline { get; init; }
}

/// <summary>One file from an agent's knowledge directory, as it will be ingested.</summary>
/// <param name="Source">The path relative to the knowledge directory, which is what a passage is cited by.</param>
/// <param name="Text">The file.</param>
public sealed record Document(string Source, string Text);

/// <summary>
/// One page from <c>knowledge/urls.yaml</c>. A page is a subscription rather than a copy:
/// what a crawler makes of it is what ends up in the knowledge base.
/// </summary>
public sealed record KnowledgePage(string Url, string Title = "", string Description = "");

/// <summary>What the video a skill captures is.</summary>
public sealed record VideoDeclaration(string Source, int MaxFrames);

/// <summary>
/// What <c>agent.yaml</c> declares: what the agent is called and what it runs on.
/// </summary>
/// <remarks>
/// A field left out leaves whatever the stored config has, so a model chosen in the
/// dashboard survives a sync that says nothing about it.
/// </remarks>
public sealed record Declaration
{
    /// <summary>What the agent is called.</summary>
    public string Name { get; init; } = "";

    /// <summary>What it is for.</summary>
    public string Description { get; init; } = "";

    /// <summary>The agent mode.</summary>
    public string Mode { get; init; } = "";

    /// <summary>The model that transcribes.</summary>
    public string Stt { get; init; } = "";

    /// <summary>The model that speaks.</summary>
    public string Tts { get; init; } = "";

    /// <summary>Null when the declaration says nothing, and empty when it turns speech-to-speech off.</summary>
    public string? Sts { get; init; }

    /// <summary>A provider-specific voice id.</summary>
    public string Voice { get; init; } = "";

    /// <summary>The model that answers.</summary>
    public string Llm { get; init; } = "";

    /// <summary>The model delegated work runs on.</summary>
    public string Subagent { get; init; } = "";

    /// <summary>The search target.</summary>
    public string Search { get; init; } = "";

    /// <summary>Said on joining, without going through the model.</summary>
    public string Greeting { get; init; } = "";

    /// <summary>Where the subagent may run code it writes.</summary>
    public string Sandbox { get; init; } = "";

    /// <summary>The plugins the agent may use.</summary>
    public IReadOnlyList<string> Plugins { get; init; } = [];

    /// <summary>Words the transcriber would otherwise get wrong.</summary>
    public IReadOnlyList<string> Keyterms { get; init; } = [];

    /// <summary>Cost labels the config carries onto every request.</summary>
    public IReadOnlyDictionary<string, string> Tags { get; init; } = new Dictionary<string, string>();

    /// <summary>Which video a skill that captures it sees.</summary>
    public VideoDeclaration? Video { get; init; }
}

/// <summary>
/// An agent written down as a directory.
/// </summary>
/// <remarks>
/// <code>
/// agents/jean/
///   agent.yaml
///   instructions.md
///   guardrail.md
///   skills/think.md
///   knowledge/pricing.md
///   knowledge/urls.yaml
/// </code>
/// </remarks>
public sealed class Folder
{
    /// <summary>What makes a directory an agent.</summary>
    public const string AgentFile = "agent.yaml";

    /// <summary>Where a directory records the fingerprint it was last synced under.</summary>
    public const string StampFile = ".agent_sync";

    /// <summary>The system prompt.</summary>
    public const string InstructionsFile = "instructions.md";

    /// <summary>The policy screening what may be asked of the agent.</summary>
    public const string GuardrailFile = "guardrail.md";

    /// <summary>The skills directory.</summary>
    public const string SkillsDir = "skills";

    /// <summary>The knowledge directory.</summary>
    public const string KnowledgeDir = "knowledge";

    /// <summary>The pages a knowledge directory is kept filled from.</summary>
    public const string KnowledgeUrlsFile = "urls.yaml";

    // Written the way Go writes it, so the + in the offset is not escaped.
    private static readonly JsonSerializerOptions StampJson = new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    // A model looks things up in prose, not in a binary.
    private static readonly HashSet<string> Readable = [".md", ".mdx", ".txt", ".rst", ".yaml", ".yml"];

    private Folder(string path, string name)
    {
        Path = path;
        Name = name;
    }

    /// <summary>The directory this was read from.</summary>
    public string Path { get; }

    /// <summary>What agent.yaml calls the agent, or the directory's own name if it does not.</summary>
    public string Name { get; private set; }

    /// <summary>agent.yaml as written, which is what its fingerprint is taken over.</summary>
    public string Source { get; private set; } = "";

    /// <summary>What agent.yaml declares.</summary>
    public Declaration Declaration { get; private set; } = new();

    /// <summary>instructions.md, or empty if there is none.</summary>
    public string Instructions { get; private set; } = "";

    /// <summary>
    /// guardrail.md, whole and unparsed. The backend parses it, so a policy this SDK has
    /// never heard of still reaches it.
    /// </summary>
    public string Guardrail { get; private set; } = "";

    /// <summary>The files in skills/, in name order.</summary>
    public IReadOnlyList<Skill> Skills { get; private set; } = [];

    /// <summary>The readable files under knowledge/, in path order.</summary>
    public IReadOnlyList<Document> Knowledge { get; private set; } = [];

    /// <summary>The pages knowledge/urls.yaml declares, in the order it lists them.</summary>
    public IReadOnlyList<KnowledgePage> KnowledgeUrls { get; private set; } = [];

    /// <summary>
    /// Where the directory's knowledge is looked up: the agent's own name, so two agents
    /// never read each other's. Empty when there is nothing to look up.
    /// </summary>
    public string KnowledgeNamespace => Knowledge.Count == 0 && KnowledgeUrls.Count == 0 ? "" : Name;

    /// <summary>
    /// Reads an agent directory.
    /// </summary>
    /// <remarks>
    /// agent.yaml is what makes a directory an agent, so it is required. Everything else is
    /// optional: a directory with only instructions.md beside it is a valid agent.
    /// </remarks>
    /// <exception cref="ConfigurationException">It is not an agent directory, or something in it cannot be read as one.</exception>
    public static Folder Load(string path)
    {
        if (!Directory.Exists(path))
        {
            throw new ConfigurationException($"{path} is not an agent directory");
        }

        var root = System.IO.Path.GetFullPath(path);
        var folder = new Folder(root, System.IO.Path.GetFileName(root.TrimEnd(System.IO.Path.DirectorySeparatorChar)));
        var declaration = System.IO.Path.Combine(root, AgentFile);
        if (!File.Exists(declaration))
        {
            throw new ConfigurationException($"{path} has no {AgentFile}, so it is not an agent directory");
        }

        var written = File.ReadAllText(declaration);
        folder.Source = written.Trim();
        folder.Declaration = Declare(written, declaration);
        if (folder.Declaration.Name != "")
        {
            folder.Name = folder.Declaration.Name;
        }
        folder.Instructions = ReadIfThere(System.IO.Path.Combine(root, InstructionsFile));
        folder.Guardrail = ReadIfThere(System.IO.Path.Combine(root, GuardrailFile));
        folder.Skills = LoadSkills(System.IO.Path.Combine(root, SkillsDir));
        folder.Knowledge = LoadKnowledge(System.IO.Path.Combine(root, KnowledgeDir));
        folder.KnowledgeUrls = LoadPages(System.IO.Path.Combine(root, KnowledgeDir, KnowledgeUrlsFile));
        return folder;
    }

    /// <summary>
    /// The agent directory called <paramref name="name"/>, or null when there is none.
    /// </summary>
    /// <remarks>
    /// <paramref name="name"/> may itself be a path. Otherwise this walks up from
    /// <paramref name="start"/>, the current directory by default, looking under
    /// <c>examples/*/</c>, then <c>agents/</c>, then for a sibling of that name, the way the
    /// Python SDK does. Only a directory holding agent.yaml counts, so a config that lives on
    /// the router and nowhere on disk is not mistaken for one that shares its name.
    /// </remarks>
    public static string? Find(string name, string? start = null)
    {
        if (IsAgent(name))
        {
            return System.IO.Path.GetFullPath(name);
        }
        if (name.Contains('/') || name.Contains('\\'))
        {
            return null;
        }

        var here = new DirectoryInfo(start ?? Directory.GetCurrentDirectory());
        if (here.Name == name && IsAgent(here.FullName))
        {
            return here.FullName;
        }
        for (; here is not null; here = here.Parent)
        {
            var examples = System.IO.Path.Combine(here.FullName, "examples");
            var candidates = Directory.Exists(examples)
                ? Directory.GetDirectories(examples).Order(StringComparer.Ordinal)
                    .Select(kind => System.IO.Path.Combine(kind, name)).ToList()
                : [];
            candidates.Add(System.IO.Path.Combine(here.FullName, "agents", name));
            candidates.Add(System.IO.Path.Combine(here.FullName, name));
            if (candidates.FirstOrDefault(IsAgent) is { } found)
            {
                return System.IO.Path.GetFullPath(found);
            }
        }
        return null;
    }

    /// <summary>
    /// A fingerprint of the directory. The Go and Python SDKs take it the same way, so a
    /// stamp any of them wrote is understood by all three.
    /// </summary>
    public string Hash() => Fingerprint(Source, Instructions, Guardrail, Skills, Knowledge, KnowledgeUrls);

    /// <summary>The fingerprint the directory was last synced under, or empty.</summary>
    public string ReadStamp()
    {
        try
        {
            var stamp = JsonSerializer.Deserialize<Stamp>(File.ReadAllText(System.IO.Path.Combine(Path, StampFile)));
            return stamp?.Hash ?? "";
        }
        catch (Exception failure) when (failure is IOException or JsonException or UnauthorizedAccessException)
        {
            return "";
        }
    }

    /// <summary>Records what was synced and when, so a second sync can do nothing.</summary>
    public void WriteStamp(string hash)
    {
        var syncedAt = DateTimeOffset.UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss'+00:00'", CultureInfo.InvariantCulture);
        File.WriteAllText(System.IO.Path.Combine(Path, StampFile),
            JsonSerializer.Serialize(new Stamp(hash, syncedAt), StampJson) + "\n");
    }

    /// <summary>
    /// The fingerprint, ported from Go's <c>fingerprint</c> byte for byte.
    /// </summary>
    internal static string Fingerprint(
        string declaration,
        string instructions,
        string guardrail,
        IEnumerable<Skill> skills,
        IEnumerable<Document> knowledge,
        IEnumerable<KnowledgePage> pages)
    {
        var text = new StringBuilder();
        text.Append(declaration).Append('\n').Append(instructions).Append('\n').Append(guardrail);
        foreach (var skill in skills.OrderBy(skill => skill.Name, StringComparer.Ordinal))
        {
            // Written the way Python prints a bool and a float, which is what keeps the
            // SDKs' fingerprints of one directory the same.
            text.Append("\nskill:").Append(skill.Name).Append('\n').Append(skill.Description).Append('\n')
                .Append(skill.Instructions).Append(skill.CaptureVideo ? "True" : "False").Append('\n');
            if (skill.Deadline is { } deadline && deadline > TimeSpan.Zero)
            {
                var seconds = deadline.TotalSeconds.ToString(CultureInfo.InvariantCulture);
                text.Append(seconds.Contains('.') ? seconds : seconds + ".0");
            }
        }
        foreach (var document in knowledge.OrderBy(document => document.Source, StringComparer.Ordinal))
        {
            text.Append("\nknowledge:").Append(document.Source).Append('\n').Append(document.Text);
        }
        foreach (var page in pages)
        {
            text.Append("\nurl:").Append(page.Url).Append('\n').Append(page.Title).Append('\n').Append(page.Description);
        }
        return Convert.ToHexStringLower(MD5.HashData(Encoding.UTF8.GetBytes(text.ToString())));
    }

    /// <summary>A map printed the way Go's <c>fmt.Sprint</c> prints one, keys sorted.</summary>
    internal static string GoMap(IReadOnlyDictionary<string, string>? map) =>
        "map[" + string.Join(' ', (map ?? new Dictionary<string, string>())
            .OrderBy(pair => pair.Key, StringComparer.Ordinal).Select(pair => $"{pair.Key}:{pair.Value}")) + "]";

    private static bool IsAgent(string path) => File.Exists(System.IO.Path.Combine(path, AgentFile));

    private static string ReadIfThere(string path) => File.Exists(path) ? File.ReadAllText(path).Trim() : "";

    /// <summary>
    /// Reads agent.yaml. A key nobody knows is refused rather than dropped, since a
    /// misspelled llm that goes quietly is a config running on a model the file does not name.
    /// </summary>
    private static Declaration Declare(string written, string path)
    {
        Written? read;
        try
        {
            // Unmatched properties are refused by default, which is the point: there is no
            // IgnoreUnmatchedProperties here, and there must not be.
            read = new DeserializerBuilder().WithDuplicateKeyChecking().Build().Deserialize<Written?>(written);
        }
        catch (YamlException failure)
        {
            throw new ConfigurationException($"{path}: {failure.InnerException?.Message ?? failure.Message}");
        }
        if (read is null)
        {
            return new Declaration();
        }

        VideoDeclaration? video = null;
        if (read.Video is { } declared)
        {
            var frames = declared.MaxFrames is null or 0 ? 1 : declared.MaxFrames.Value;
            if (frames is < 1 or > 8)
            {
                throw new ConfigurationException($"{path}: video.max_frames must be an integer from 1 to 8");
            }
            video = new VideoDeclaration(declared.Source ?? "", frames);
        }
        return new Declaration
        {
            Name = read.Name ?? "",
            Description = read.Description ?? "",
            Mode = read.Mode ?? "",
            Stt = read.Stt ?? "",
            Tts = read.Tts ?? "",
            Sts = read.Sts,
            Voice = read.Voice ?? "",
            Llm = read.Llm ?? "",
            Subagent = read.Subagent ?? "",
            Search = read.Search ?? "",
            Greeting = read.Greeting ?? "",
            Sandbox = read.Sandbox ?? "",
            Plugins = read.Plugins ?? [],
            Keyterms = read.Keyterms ?? [],
            Tags = read.Tags ?? [],
            Video = video,
        };
    }

    private static List<Skill> LoadSkills(string path)
    {
        if (!Directory.Exists(path))
        {
            return [];
        }
        var skills = new List<Skill>();
        foreach (var file in Directory.GetFiles(path, "*.md").Order(StringComparer.Ordinal))
        {
            try
            {
                skills.Add(ParseSkill(System.IO.Path.GetFileNameWithoutExtension(file), File.ReadAllText(file)));
            }
            catch (ConfigurationException failure)
            {
                throw new ConfigurationException($"{file}: {failure.Message}");
            }
        }
        return skills;
    }

    /// <summary>
    /// Reads a skill file: frontmatter between --- lines, then the instructions. The keys are
    /// name, description, capture_video and deadline, which is a Go duration or seconds.
    /// </summary>
    internal static Skill ParseSkill(string name, string content)
    {
        string description = "";
        var captureVideo = false;
        TimeSpan? deadline = null;

        var (frontmatter, body, found) = CutFrontmatter(content);
        if (found)
        {
            foreach (var raw in frontmatter.Split('\n'))
            {
                var line = raw.Trim();
                if (line == "" || line.StartsWith('#'))
                {
                    continue;
                }
                var colon = line.IndexOf(':');
                if (colon < 0)
                {
                    throw new ConfigurationException($"\"{line}\" is not a key and a value");
                }
                var value = line[(colon + 1)..].Trim().Trim('"', '\'');
                switch (line[..colon].Trim())
                {
                    case "name":
                        name = value;
                        break;
                    case "description":
                        description = value;
                        break;
                    case "capture_video":
                        captureVideo = value switch
                        {
                            "true" => true,
                            "false" => false,
                            _ => throw new ConfigurationException("capture_video must be true or false"),
                        };
                        break;
                    case "deadline":
                        deadline = ParseDeadline(value);
                        break;
                }
            }
        }

        var instructions = body.Trim();
        if (description == "")
        {
            throw new ConfigurationException("a skill needs a description, since it is all the fast model sees");
        }
        if (instructions == "")
        {
            throw new ConfigurationException("a skill needs instructions, since they are what the subagent answers under");
        }
        return new Skill
        {
            Name = name,
            Description = description,
            Instructions = instructions,
            CaptureVideo = captureVideo,
            Deadline = deadline,
        };
    }

    /// <summary>A Go duration such as <c>30s</c> or <c>1m30s</c>, or a bare number of seconds.</summary>
    internal static TimeSpan ParseDeadline(string value)
    {
        if (double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var seconds))
        {
            return TimeSpan.FromSeconds(seconds);
        }

        var total = TimeSpan.Zero;
        var rest = value.AsSpan();
        if (rest.IsEmpty)
        {
            throw new ConfigurationException($"\"{value}\" is not a deadline");
        }
        while (!rest.IsEmpty)
        {
            var digits = 0;
            while (digits < rest.Length && (char.IsAsciiDigit(rest[digits]) || rest[digits] == '.'))
            {
                digits++;
            }
            var units = digits;
            while (units < rest.Length && !char.IsAsciiDigit(rest[units]) && rest[units] != '.')
            {
                units++;
            }
            if (digits == 0 || !double.TryParse(rest[..digits], NumberStyles.Float, CultureInfo.InvariantCulture, out var amount))
            {
                throw new ConfigurationException($"\"{value}\" is not a deadline");
            }
            total += rest[digits..units] switch
            {
                "ns" => TimeSpan.FromTicks((long)(amount / 100)),
                "us" or "µs" => TimeSpan.FromTicks((long)(amount * 10)),
                "ms" => TimeSpan.FromMilliseconds(amount),
                "s" => TimeSpan.FromSeconds(amount),
                "m" => TimeSpan.FromMinutes(amount),
                "h" => TimeSpan.FromHours(amount),
                _ => throw new ConfigurationException($"\"{value}\" is not a deadline"),
            };
            rest = rest[units..];
        }
        return total;
    }

    private static (string Frontmatter, string Body, bool Found) CutFrontmatter(string content)
    {
        var trimmed = content.TrimStart('\ufeff', ' ', '\t', '\r', '\n');
        if (!trimmed.StartsWith("---", StringComparison.Ordinal))
        {
            return ("", content, false);
        }
        var rest = trimmed[3..].TrimStart('\r', '\n');
        var end = rest.IndexOf("\n---", StringComparison.Ordinal);
        if (end < 0)
        {
            return ("", content, false);
        }
        return (rest[..end], rest[(end + 4)..].TrimStart('-', '\r', '\n'), true);
    }

    private static List<Document> LoadKnowledge(string path)
    {
        if (!Directory.Exists(path))
        {
            return [];
        }
        // Only the declaration at the root is not a document; deeper, urls.yaml is one like
        // any other.
        var declaration = System.IO.Path.Combine(path, KnowledgeUrlsFile);
        var documents = new List<Document>();
        foreach (var file in Directory.EnumerateFiles(path, "*", SearchOption.AllDirectories))
        {
            if (!Readable.Contains(System.IO.Path.GetExtension(file).ToLowerInvariant()) || file == declaration)
            {
                continue;
            }
            var text = File.ReadAllText(file);
            if (text.Trim() == "")
            {
                continue;
            }
            documents.Add(new Document(System.IO.Path.GetRelativePath(path, file).Replace('\\', '/'), text));
        }
        documents.Sort((one, other) => string.CompareOrdinal(one.Source, other.Source));
        return documents;
    }

    /// <summary>
    /// Reads the pages a knowledge base is kept filled from. A bad url or a key nobody knows
    /// is refused here, before anything is written.
    /// </summary>
    private static List<KnowledgePage> LoadPages(string path)
    {
        if (!File.Exists(path))
        {
            return [];
        }

        var stream = new YamlStream();
        try
        {
            stream.Load(new StringReader(File.ReadAllText(path)));
        }
        catch (YamlException failure)
        {
            throw new ConfigurationException($"{path}: {failure.Message}");
        }
        if (stream.Documents.Count == 0 || stream.Documents[0].RootNode is YamlScalarNode { Value: null or "" })
        {
            return [];
        }
        if (stream.Documents[0].RootNode is not YamlSequenceNode listed)
        {
            throw new ConfigurationException($"{path} should list pages");
        }

        var pages = new List<KnowledgePage>();
        foreach (var item in listed)
        {
            var page = item switch
            {
                YamlScalarNode scalar => new KnowledgePage((scalar.Value ?? "").Trim()),
                YamlMappingNode mapping => PageOf(mapping, path),
                _ => throw new ConfigurationException($"{path}: a page is a url, or a mapping naming one"),
            };
            if (!page.Url.StartsWith("http://", StringComparison.Ordinal) && !page.Url.StartsWith("https://", StringComparison.Ordinal))
            {
                throw new ConfigurationException($"{path}: \"{page.Url}\" is not an http or https url");
            }
            pages.Add(page);
        }
        return pages;
    }

    private static KnowledgePage PageOf(YamlMappingNode mapping, string path)
    {
        string url = "", title = "", description = "";
        foreach (var (key, value) in mapping.Children)
        {
            var said = value is YamlScalarNode scalar ? (scalar.Value ?? "").Trim()
                : throw new ConfigurationException($"{path}: what a page says is a string");
            switch ((key as YamlScalarNode)?.Value)
            {
                case "url":
                    url = said;
                    break;
                case "title":
                    title = said;
                    break;
                case "description":
                    description = said;
                    break;
                default:
                    throw new ConfigurationException(
                        $"{path}: \"{(key as YamlScalarNode)?.Value}\" is not something a page says; url, title and description are");
            }
        }
        return new KnowledgePage(url, title, description);
    }

    private sealed record Stamp(
        [property: JsonPropertyName("hash")] string Hash,
        [property: JsonPropertyName("synced_at")] string SyncedAt);

    private sealed class Written
    {
        [YamlMember(Alias = "name")] public string? Name { get; set; }
        [YamlMember(Alias = "description")] public string? Description { get; set; }
        [YamlMember(Alias = "mode")] public string? Mode { get; set; }
        [YamlMember(Alias = "stt")] public string? Stt { get; set; }
        [YamlMember(Alias = "tts")] public string? Tts { get; set; }
        [YamlMember(Alias = "sts")] public string? Sts { get; set; }
        [YamlMember(Alias = "voice")] public string? Voice { get; set; }
        [YamlMember(Alias = "llm")] public string? Llm { get; set; }
        [YamlMember(Alias = "subagent")] public string? Subagent { get; set; }
        [YamlMember(Alias = "search")] public string? Search { get; set; }
        [YamlMember(Alias = "greeting")] public string? Greeting { get; set; }
        [YamlMember(Alias = "sandbox")] public string? Sandbox { get; set; }
        [YamlMember(Alias = "plugins")] public List<string>? Plugins { get; set; }
        [YamlMember(Alias = "keyterms")] public List<string>? Keyterms { get; set; }
        [YamlMember(Alias = "tags")] public Dictionary<string, string>? Tags { get; set; }
        [YamlMember(Alias = "video")] public WrittenVideo? Video { get; set; }
    }

    private sealed class WrittenVideo
    {
        [YamlMember(Alias = "source")] public string? Source { get; set; }
        [YamlMember(Alias = "max_frames")] public int? MaxFrames { get; set; }
    }
}
