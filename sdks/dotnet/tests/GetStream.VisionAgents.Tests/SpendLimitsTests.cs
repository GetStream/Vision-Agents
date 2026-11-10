using System.Text.Json;
using System.Text.Json.Nodes;
using GetStream.VisionAgents.Models;

namespace GetStream.VisionAgents.Tests;

public sealed class SpendLimitsTests
{
    [Fact]
    public void AnAmountIsHeldExactlyInMicros()
    {
        Assert.Equal(20_000_000_000, Usd.Dollars(20_000).Micros);
        Assert.Equal(30_000, Usd.Cents(3).Micros);
        Assert.Equal(12_500_000, Usd.Parse("$12.50").Micros);
        Assert.Equal(300, Usd.Parse("0.000300").Micros);
        Assert.Equal(-3_250_000, Usd.Parse("-3.25").Micros);
        Assert.Equal(Usd.Parse("0.3"), Usd.FromDecimal(0.1m) + Usd.FromDecimal(0.2m));
        Assert.Equal(12.5m, Usd.FromMicros(12_500_000).ToDecimal());
        Assert.Equal(Usd.Dollars(25), Usd.Dollars(10) + (3 * Usd.Dollars(5)));
        Assert.Equal("-$9223372036854.775808", Usd.FromMicros(long.MinValue).ToString());
    }

    [Theory]
    [InlineData(20_000_000_000, "$20000.00")]
    [InlineData(12_500_000, "$12.50")]
    [InlineData(300, "$0.0003")]
    [InlineData(-3_250_000, "-$3.25")]
    public void AnAmountPrintsAsDollarsAndReadsBack(long micros, string written)
    {
        Assert.Equal(written, Usd.FromMicros(micros).ToString());
        Assert.Equal(micros, Usd.Parse(written).Micros);
    }

    [Theory]
    [InlineData("")]
    [InlineData("12.")]
    [InlineData("1,000")]
    [InlineData("1e3")]
    [InlineData("0.0000001")]
    [InlineData("9223372036855")]
    [InlineData("twelve")]
    public void WhatIsNotAnAmountTheRouterCanCountIsRefused(string written)
    {
        Assert.False(Usd.TryParse(written, out _));
        Assert.Throws<FormatException>(() => Usd.Parse(written));
    }

    [Fact]
    public void ADecimalFinerThanAMicroIsRefused()
    {
        Assert.Throws<FormatException>(() => Usd.FromDecimal(0.0000001m));
        Assert.Throws<OverflowException>(() => Usd.Dollars(long.MaxValue));
    }

    [Fact]
    public async Task ALimitIsCreatedWithTheRouterSpelling()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/spend/limits", 201, Limit(1));
        using var client = Fixtures.Client(router);
        var limits = client.SpendLimits;
        var cancel = TestContext.Current.CancellationToken;

        await limits.CreateAsync(new LimitOptions { Subject = LimitSubject.Organization, Interval = LimitInterval.Monthly, Amount = Usd.Dollars(20_000) }, cancel);
        await limits.CreateAsync(new LimitOptions
        {
            Subject = LimitSubject.ApiKey,
            AppId = "1234",
            ApiKey = "mz7kq2xa9b",
            Interval = LimitInterval.Daily,
            Amount = Usd.Dollars(10),
            Level = LimitLevel.ApiKey,
            Note = "CI key",
        }, cancel);
        await limits.CreateAsync(new LimitOptions { Subject = LimitSubject.Team, Team = "finance", AppId = "1234", Interval = LimitInterval.Biweekly, Amount = Usd.Cents(1) }, cancel);

        var bodies = router.To("POST", "/v1/spend/limits").Select(seen => seen.Body!.ToJsonString()).ToList();
        Assert.Equal("""{"interval":"monthly","limit_micros":20000000000,"subject":"organization"}""", bodies[0]);
        Assert.Equal("""{"api_key":"mz7kq2xa9b","app_id":"1234","interval":"daily","level":"api_key","limit_micros":10000000,"note":"CI key","subject":"api_key"}""", bodies[1]);
        Assert.Equal("""{"app_id":"1234","interval":"biweekly","limit_micros":10000,"subject":"app_team","team":"finance"}""", bodies[2]);
    }

    [Fact]
    public async Task EveryPageOfLimitsIsReadWithTheFilters()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/spend/limits", seen => new Reply(200, seen.Query.ContainsKey("cursor")
            ? new { items = new[] { Limit(2) }, has_more = false }
            : new { items = new[] { Limit(1) }, has_more = true, next_cursor = "c2" }));
        using var client = Fixtures.Client(router);

        var limits = await client.SpendLimits.ListAsync(new LimitQuery
        {
            Subject = LimitSubject.ApiKey,
            AppId = "1234",
            ApiKey = "mz7kq2xa9b",
            Interval = LimitInterval.Daily,
            Level = LimitLevel.ApiKey,
        }, TestContext.Current.CancellationToken);

        Assert.Equal([1L, 2L], limits.Select(limit => limit.Id));
        var pages = router.To("GET", "/v1/spend/limits");
        Assert.Equal(new Dictionary<string, string>
        {
            ["subject"] = "api_key",
            ["level"] = "api_key",
            ["interval"] = "daily",
            ["app_id"] = "1234",
            ["api_key_id"] = "mz7kq2xa9b",
            ["limit"] = "200",
        }, pages[0].Query);
        Assert.Equal("c2", pages[1].Query["cursor"]);
        Assert.False(pages[1].Query.ContainsKey("api_key"));
    }

    [Fact]
    public async Task ALimitIsReadChangedAndDeletedByItsId()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/spend/limits/7", 200, Limit(7));
        router.On("PATCH", "/v1/spend/limits/7", 200, Limit(7));
        router.On("DELETE", "/v1/spend/limits/7", 204);
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;

        var read = await client.SpendLimits.GetAsync(7, cancel);
        await client.SpendLimits.SetAmountAsync(7, Usd.Dollars(25), cancel);
        await client.SpendLimits.SetNoteAsync(7, "", cancel);
        await client.SpendLimits.DeleteAsync(7, cancel);

        Assert.Equal((7L, Usd.Dollars(50)), (read.Id, Usd.FromMicros(read.LimitMicros)));
        var changes = router.To("PATCH", "/v1/spend/limits/7").Select(seen => seen.Body!.ToJsonString()).ToList();
        Assert.Equal(["""{"limit_micros":25000000}""", """{"note":""}"""], changes);
        router.Only("DELETE", "/v1/spend/limits/7");
    }

    [Fact]
    public async Task AResetNamesTheSubjectTheReasonAndTheWindows()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("POST", "/v1/spend/limits/usage/reset", 200, new { subject = "user", app_id = "1234", user_id = "user-42", windows = Array.Empty<object>() });
        using var client = Fixtures.Client(router);

        var usage = await client.SpendLimits.ResetAsync(new LimitReset
        {
            Subject = LimitSubject.User,
            AppId = "1234",
            UserId = "user-42",
            Intervals = [LimitInterval.Weekly],
            Reason = "refunded a runaway session",
        }, TestContext.Current.CancellationToken);

        Assert.Equal("user-42", usage.UserId);
        Assert.Equal("""{"app_id":"1234","intervals":["weekly"],"reason":"refunded a runaway session","subject":"user","user_id":"user-42"}""",
            router.Only("POST", "/v1/spend/limits/usage/reset").Body!.ToJsonString());
    }

    [Fact]
    public async Task WhatTheRouterWouldRefuseIsRefusedBeforeAnythingIsSent()
    {
        await using var router = await TestRouter.StartAsync();
        using var client = Fixtures.Client(router);
        var limits = client.SpendLimits;
        var cancel = TestContext.Current.CancellationToken;
        LimitOptions Daily(LimitSubject subject) => new() { Subject = subject, Interval = LimitInterval.Daily, Amount = Usd.Dollars(1) };

        await Assert.ThrowsAsync<ConfigurationException>(() => limits.CreateAsync(Daily(LimitSubject.Team), cancel));
        await Assert.ThrowsAsync<ConfigurationException>(() => limits.CreateAsync(Daily(LimitSubject.AppTeam) with { Team = "finance" }, cancel));
        await Assert.ThrowsAsync<ConfigurationException>(() => limits.CreateAsync(Daily(LimitSubject.User) with { Team = "finance" }, cancel));
        await Assert.ThrowsAsync<ConfigurationException>(() => limits.CreateAsync(Daily(LimitSubject.Team) with { Team = "fin ance" }, cancel));
        var team = await Assert.ThrowsAsync<ConfigurationException>(() => limits.CreateAsync(Daily(LimitSubject.Category) with { Category = "Team:finance" }, cancel));
        Assert.Contains("names a team", team.Message);
        await Assert.ThrowsAsync<ConfigurationException>(() => limits.CreateAsync(Daily(LimitSubject.App) with { Amount = -Usd.Cents(1) }, cancel));
        await Assert.ThrowsAsync<ConfigurationException>(() => limits.SetAmountAsync(1, -Usd.Cents(1), cancel));
        await Assert.ThrowsAsync<ConfigurationException>(() => limits.ResetAsync(new LimitReset { Subject = LimitSubject.User, UserId = "user-42", Reason = " " }, cancel));
        await Assert.ThrowsAsync<ConfigurationException>(() => limits.ListAsync(new LimitQuery { Category = "team:finance" }, cancel));
        Assert.Empty(router.Requests);
    }

    [Fact]
    public async Task TeamsAreListedWithinAnApp()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/teams", 200, new[] { new { name = "finance", apps = new[] { "1234" }, limits = new[] { Limit(3, team: "finance") }, spent = Array.Empty<object>() } });
        using var client = Fixtures.Client(router);

        var teams = await client.Teams.ListAsync("1234", TestContext.Current.CancellationToken);

        Assert.Equal("finance", Assert.Single(teams).Name);
        Assert.Equal("1234", router.Only("GET", "/v1/teams").Query["app_id"]);
    }

    [Fact]
    public async Task ATeamsLimitIsCreatedWhenItHasNone()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/spend/limits", 200, new { items = new[] { Limit(3, team: "finance", appId: "1234") }, has_more = false });
        router.On("POST", "/v1/spend/limits", 201, Limit(4, team: "finance"));
        using var client = Fixtures.Client(router);

        var limit = await client.Teams.SetLimitAsync(new TeamLimit { Team = "finance", Interval = LimitInterval.Monthly, Amount = Usd.Dollars(8_000) },
            TestContext.Current.CancellationToken);

        Assert.Equal(4, limit.Id);
        var looked = router.Only("GET", "/v1/spend/limits").Query;
        Assert.Equal(("team", "finance", "monthly"), (looked["subject"], looked["team"], looked["interval"]));
        Assert.Equal("""{"interval":"monthly","limit_micros":8000000000,"subject":"team","team":"finance"}""",
            router.Only("POST", "/v1/spend/limits").Body!.ToJsonString());
    }

    [Fact]
    public async Task ATeamsLimitIsChangedWhenItHasOneEvenIfItWasCreatedMeanwhile()
    {
        await using var router = await TestRouter.StartAsync();
        var looked = 0;
        router.On("GET", "/v1/spend/limits", _ => new Reply(200, ++looked == 1
            ? new { items = Array.Empty<object>(), has_more = false }
            : new { items = new[] { Limit(5, team: "finance", appId: "1234") }, has_more = false }));
        router.On("POST", "/v1/spend/limits", 409, Fixtures.Failure("conflict", "limit_exists", "the team has a monthly limit"));
        router.On("PATCH", "/v1/spend/limits/5", 200, Limit(5, team: "finance", appId: "1234"));
        using var client = Fixtures.Client(router);

        await client.Teams.SetLimitAsync(new TeamLimit
        {
            Team = "finance",
            AppId = "1234",
            Interval = LimitInterval.Monthly,
            Amount = Usd.Dollars(8_000),
            Note = "Q4",
        }, TestContext.Current.CancellationToken);

        Assert.Equal("app_team", router.Only("POST", "/v1/spend/limits").Body.Text("subject"));
        Assert.Equal("""{"limit_micros":8000000000,"note":"Q4"}""", router.Only("PATCH", "/v1/spend/limits/5").Body!.ToJsonString());
    }

    [Fact]
    public async Task ATeamsLimitIsRemovedOnlyWhenItHasOne()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/spend/limits", seen => new Reply(200, seen.Query.ContainsKey("app_id")
            ? new { items = new[] { Limit(6, team: "finance", appId: "1234") }, has_more = false }
            : new { items = new[] { Limit(9, team: "finance", appId: "1234") }, has_more = false }));
        router.On("DELETE", "/v1/spend/limits/6", 404, Fixtures.Failure("not_found", "not_found", "no such limit"));
        using var client = Fixtures.Client(router);
        var cancel = TestContext.Current.CancellationToken;

        await client.Teams.RemoveLimitAsync("finance", LimitInterval.Monthly, "1234", cancel);
        await client.Teams.RemoveLimitAsync("finance", LimitInterval.Monthly, cancellationToken: cancel);

        router.Only("DELETE", "/v1/spend/limits/6");
        Assert.Empty(router.To("DELETE", "/v1/spend/limits/9"));
        await Assert.ThrowsAsync<ConfigurationException>(() => client.Teams.RemoveLimitAsync("team:finance", LimitInterval.Daily, cancellationToken: cancel));
    }

    [Fact]
    public async Task ATeamAndCategoriesAreSignedIntoEveryTokenMinted()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/teams", 200, Array.Empty<object>());
        var options = new VisionAgentsOptions
        {
            Url = router.Url,
            ApiKey = "key",
            ApiSecret = Fixtures.Secret,
            Team = "finance",
            UsageCategories = ["project:onboarding", "cost-center.7", "project:onboarding"],
        };
        using var keyed = new VisionAgentsClient(options);
        using var proxied = new VisionAgentsClient(options with { Authenticate = true, UserId = "user-42" });
        var cancel = TestContext.Current.CancellationToken;

        await keyed.Teams.ListAsync(cancellationToken: cancel);
        await proxied.Teams.ListAsync(cancellationToken: cancel);

        var (server, user) = (router.To("GET", "/v1/teams")[0], router.To("GET", "/v1/teams")[1]);
        foreach (var claims in new[] { Fixtures.Claims(server.Headers["Authorization"]), Fixtures.Claims(user.Headers["Authorization"]) })
        {
            Assert.Equal("finance", claims.Text("team"));
            Assert.Equal("""["cost-center.7","project:onboarding"]""", claims["usage_categories"]!.ToJsonString());
        }
        Assert.Equal("user-42", Fixtures.Claims(user.Headers["Authorization"]).Text("user_id"));
        Assert.Equal("cost-center.7,project:onboarding,team:finance", server.Headers[UsageCategories.Header]);
        Assert.False(user.Headers.ContainsKey(UsageCategories.Header));
    }

    [Fact]
    public async Task ARouterThatTrustsACustomerIsToldTheTeamInAHeader()
    {
        await using var router = await TestRouter.StartAsync();
        router.On("GET", "/v1/teams", 200, Array.Empty<object>());
        using var client = new VisionAgentsClient(new VisionAgentsOptions { Url = router.Url, CustomerId = "examples", Team = "finance" });
        using var untagged = Fixtures.Client(router);

        await client.Teams.ListAsync(cancellationToken: TestContext.Current.CancellationToken);
        await untagged.Teams.ListAsync(cancellationToken: TestContext.Current.CancellationToken);

        var seen = router.To("GET", "/v1/teams");
        Assert.Equal("team:finance", seen[0].Headers[UsageCategories.Header]);
        Assert.False(seen[1].Headers.ContainsKey(UsageCategories.Header));
    }

    [Theory]
    [InlineData("fin ance", null)]
    [InlineData("a-team-name-that-goes-on-and-on-well-past-the-fifty-nine-character-limit", null)]
    [InlineData("team:finance", null)]
    [InlineData(null, "TEAM:finance")]
    [InlineData(null, "")]
    [InlineData(null, "project onboarding")]
    public void ATeamOrCategoryTheRouterWouldRefuseIsRefused(string? team, string? category)
    {
        Assert.Throws<ConfigurationException>(() => new VisionAgentsClient(new VisionAgentsOptions
        {
            Url = "http://router",
            CustomerId = "examples",
            Team = team,
            UsageCategories = category is null ? null : [category],
        }));
    }

    [Fact]
    public void AtMostTenCategoriesTheTeamIncluded()
    {
        var nine = Enumerable.Range(0, 9).Select(index => $"c{index}").ToList();
        var options = new VisionAgentsOptions { Url = "http://router", CustomerId = "examples", Team = "finance" };

        _ = new VisionAgentsClient(options with { UsageCategories = [.. nine, "c0"] });
        var refused = Assert.Throws<ConfigurationException>(() => new VisionAgentsClient(options with { UsageCategories = [.. nine, "c9"] }));
        Assert.Contains("these are 11", refused.Message);
    }

    [Fact]
    public void ATokenHandedInCannotBeGivenATeam()
    {
        var refused = Assert.Throws<ConfigurationException>(() => new VisionAgentsClient(new VisionAgentsOptions
        {
            Url = "http://router",
            ApiKey = "key",
            Token = "user-token",
            Team = "finance",
        }));
        Assert.Contains("token handed in", refused.Message);
        Assert.Throws<ConfigurationException>(() => new VisionAgentsClient(new VisionAgentsOptions
        {
            Url = "http://router",
            ApiKey = "key",
            Token = "user-token",
            UsageCategories = ["project:onboarding"],
        }));
    }

    [Fact]
    public async Task ASessionIsOpenedUnderATeamOnlyWhenOneIsNamed()
    {
        using var client = new VisionAgentsClient(new VisionAgentsOptions { Url = "http://router", CustomerId = "examples" });
        await using var agent = new Agent(new AgentOptions { Config = "support", Client = client });

        var teamed = agent.Request(null, false, new SessionOptions { Team = "finance" }, null, false);
        var plain = agent.Request(null, false, new SessionOptions(), null, false);

        Assert.Equal("finance", JsonSerializer.SerializeToNode(teamed, Json.Options).Text("team"));
        Assert.False(JsonSerializer.SerializeToNode(plain, Json.Options)!.AsObject().ContainsKey("team"));
    }

    private static object Limit(long id, string? team = null, string? appId = null) => new
    {
        id,
        subject = team is null ? "app" : appId is null ? "team" : "app_team",
        level = appId is null && team is not null ? "organization" : "app",
        interval = "monthly",
        limit_micros = 50_000_000,
        effective_limit_micros = 50_000_000,
        team,
        app_id = appId ?? (team is null ? "1234" : null),
        created_at = "2026-09-24T10:00:00Z",
        updated_at = "2026-09-24T10:00:00Z",
        updated_by = "server",
    };
}
