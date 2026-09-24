namespace GetStream.VisionAgents.Tests;

public sealed class FolderTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "va-folder-" + Guid.NewGuid().ToString("N"), "jean");

    public void Dispose()
    {
        if (Directory.Exists(Path.GetDirectoryName(_root)))
        {
            Directory.Delete(Path.GetDirectoryName(_root)!, recursive: true);
        }
    }

    [Fact]
    public void ADirectoryIsReadAsInstructionsSkillsAndKnowledge()
    {
        Write("agent.yaml", "name: jean\n");
        Write("instructions.md", "You are Jean.\n");
        Write("skills/think.md", "---\ndescription: Work something out before answering\ndeadline: 30s\n---\nTake your time and reason it through.\n");
        Write("knowledge/pricing.md", "# Pricing\n\nA call costs a penny.\n");

        var folder = Folder.Load(_root);

        Assert.Equal("jean", folder.Name);
        Assert.Equal("You are Jean.", folder.Instructions);
        var skill = Assert.Single(folder.Skills);
        Assert.Equal("think", skill.Name);
        Assert.Equal("Work something out before answering", skill.Description);
        Assert.Equal(TimeSpan.FromSeconds(30), skill.Deadline);
        Assert.Equal("Take your time and reason it through.", skill.Instructions);
        Assert.Equal("pricing.md", Assert.Single(folder.Knowledge).Source);
        Assert.Equal("jean", folder.KnowledgeNamespace);
    }

    [Fact]
    public void ADirectoryWithOnlyInstructionsIsAnAgent()
    {
        Write("agent.yaml", "");
        Write("instructions.md", "Say little.\n");

        var folder = Folder.Load(_root);

        Assert.Empty(folder.Skills);
        Assert.Empty(folder.Knowledge);
        Assert.Equal("", folder.KnowledgeNamespace);
    }

    [Fact]
    public void NestedKnowledgeKeepsThePathItWasFoundAt()
    {
        Write("agent.yaml", "name: jean\n");
        Write("knowledge/reference/api.md", "# API\n\nthe endpoints\n");
        Write("knowledge/logo.png", "not a document");
        Write("knowledge/empty.md", "   \n");

        Assert.Equal("reference/api.md", Assert.Single(Folder.Load(_root).Knowledge).Source);
    }

    [Fact]
    public void DeclaredPagesAreReadWithoutBeingIngested()
    {
        Write("agent.yaml", "name: jean\n");
        Write("knowledge/pricing.md", "# Pricing\n\nA call costs a penny.\n");
        Write("knowledge/urls.yaml", "- https://example.com/pricing\n- url: https://example.com/plans\n  title: Plans\n  description: What each plan includes.\n");
        Write("knowledge/reference/urls.yaml", "the urls we used to have\n");

        var folder = Folder.Load(_root);

        Assert.Equal(["pricing.md", "reference/urls.yaml"], folder.Knowledge.Select(document => document.Source));
        Assert.Equal(
            [new KnowledgePage("https://example.com/pricing"), new KnowledgePage("https://example.com/plans", "Plans", "What each plan includes.")],
            folder.KnowledgeUrls);
    }

    [Fact]
    public void ADirectoryWithOnlyPagesHasSomewhereToLookThingsUpIn()
    {
        Write("agent.yaml", "name: jean\n");
        Write("knowledge/urls.yaml", "- https://example.com/pricing\n");

        var folder = Folder.Load(_root);

        Assert.Empty(folder.Knowledge);
        Assert.Equal("jean", folder.KnowledgeNamespace);
    }

    [Theory]
    [InlineData("- example.com/pricing\n")]
    [InlineData("- url: https://example.com/plans\n  heading: Plans\n")]
    [InlineData("- [https://example.com/plans]\n")]
    public void APageThatCannotBeFetchedOrDescribedIsRefused(string declaration)
    {
        Write("agent.yaml", "name: jean\n");
        Write("knowledge/urls.yaml", declaration);

        Assert.Throws<ConfigurationException>(() => Folder.Load(_root));
    }

    [Fact]
    public void ASkillWithoutADescriptionIsRefused()
    {
        Write("agent.yaml", "name: jean\n");
        Write("skills/think.md", "Just a body, with nothing saying when to use it.\n");

        Assert.Throws<ConfigurationException>(() => Folder.Load(_root));
    }

    [Fact]
    public void ASkillCanBeNamedSomethingOtherThanItsFile()
    {
        Write("agent.yaml", "name: jean\n");
        Write("skills/01-think.md", "---\nname: think\ndescription: Work something out\ncapture_video: true\n---\nReason it through.\n");

        var skill = Assert.Single(Folder.Load(_root).Skills);

        Assert.Equal("think", skill.Name);
        Assert.True(skill.CaptureVideo);
        Assert.Null(skill.Deadline);
    }

    [Theory]
    [InlineData("30s", 30_000)]
    [InlineData("1m30s", 90_000)]
    [InlineData("250ms", 250)]
    [InlineData("2.5", 2_500)]
    public void ADeadlineIsAGoDurationOrSeconds(string written, int milliseconds)
    {
        Assert.Equal(TimeSpan.FromMilliseconds(milliseconds), Folder.ParseDeadline(written));
    }

    [Fact]
    public void ADirectoryWithoutADeclarationIsNotAnAgent()
    {
        Write("instructions.md", "You are Jean.\n");

        Assert.Throws<ConfigurationException>(() => Folder.Load(_root));
    }

    [Fact]
    public void TheDeclarationSaysWhatTheAgentIsCalledAndRunsOn()
    {
        Write("agent.yaml", "name: receptionist\nllm: openai/gpt-5.6\nsts: \"\"\nkeyterms: [Vision Agents]\nvideo:\n  source: camera\n");

        var folder = Folder.Load(_root);

        Assert.Equal("receptionist", folder.Name);
        Assert.Equal("openai/gpt-5.6", folder.Declaration.Llm);
        Assert.Equal(["Vision Agents"], folder.Declaration.Keyterms);
        Assert.Equal("", folder.Declaration.Sts);
        Assert.Equal(new VideoDeclaration("camera", 1), folder.Declaration.Video);
    }

    [Fact]
    public void SayingNothingAboutSpeechToSpeechIsNotTurningItOff()
    {
        Write("agent.yaml", "name: jean\n");

        Assert.Null(Folder.Load(_root).Declaration.Sts);
    }

    [Theory]
    [InlineData("name: jean\nlmm: openai/gpt-5.6\n")]
    [InlineData("video:\n  max_frames: 9\n")]
    [InlineData("keyterms: Vision Agents\n")]
    public void ADeclarationKeyNobodyKnowsIsRefused(string declaration)
    {
        Write("agent.yaml", declaration);

        Assert.Throws<ConfigurationException>(() => Folder.Load(_root));
    }

    [Fact]
    public void ADirectoryHashesTheWayTheGoAndPythonSdksHashIt()
    {
        Write("agent.yaml", "name: jean\nllm: openai/gpt-5.6\n");
        Write("instructions.md", "You are Jean.\n");
        Write("skills/think.md", "---\ndescription: Work it out\ndeadline: 30s\n---\nReason it through.\n");
        Write("knowledge/pricing.md", "# Pricing\n\nA penny.\n");

        var folder = Folder.Load(_root);
        Assert.Equal("02a7b2c8428f31e3a2b93ca2f5a6ec70", folder.Hash());

        Write("knowledge/urls.yaml", "- https://example.com/plans\n");
        Assert.NotEqual(folder.Hash(), Folder.Load(_root).Hash());
    }

    [Fact]
    public void AStampRecordsTheHashAndWhenItWasSynced()
    {
        Write("agent.yaml", "name: jean\n");
        var folder = Folder.Load(_root);
        Assert.Equal("", folder.ReadStamp());

        folder.WriteStamp("abc");

        Assert.Equal("abc", folder.ReadStamp());
        Assert.Matches("""^\{"hash":"abc","synced_at":"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00"\}\n$""",
            File.ReadAllText(Path.Combine(_root, Folder.StampFile)));
    }

    [Fact]
    public void SomethingThatIsNotADirectoryIsNotAnAgent()
    {
        Write("jean.md", "You are Jean.");

        Assert.Throws<ConfigurationException>(() => Folder.Load(Path.Combine(_root, "jean.md")));
    }

    [Fact]
    public void AnAgentIsFoundByNameUnderAgentsOrExamples()
    {
        var workspace = Path.GetDirectoryName(_root)!;
        Directory.CreateDirectory(Path.Combine(workspace, "agents", "support"));
        File.WriteAllText(Path.Combine(workspace, "agents", "support", "agent.yaml"), "");
        Directory.CreateDirectory(Path.Combine(workspace, "examples", "voice", "docs"));
        File.WriteAllText(Path.Combine(workspace, "examples", "voice", "docs", "agent.yaml"), "");
        var deeper = Directory.CreateDirectory(Path.Combine(workspace, "src", "app")).FullName;

        Assert.Equal(Path.Combine(workspace, "agents", "support"), Folder.Find("support", deeper));
        Assert.Equal(Path.Combine(workspace, "examples", "voice", "docs"), Folder.Find("docs", deeper));
        Assert.Null(Folder.Find("stored-only", deeper));
    }

    private void Write(string name, string content)
    {
        var path = Path.Combine(_root, name);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content);
    }
}
