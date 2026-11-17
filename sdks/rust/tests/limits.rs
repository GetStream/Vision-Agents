mod support;

use axum::http::Method;
use serde_json::{Value, json};
use support::{Server, refusal};
use vision_agents::types::{LimitInterval, LimitLevel, LimitSubject};
use vision_agents::{Error, Limit, LimitFilter, Reset, TeamLimit, Usd};

/// A limit as the router answers one.
fn limit(id: i64, subject: &str, interval: &str, micros: i64, extra: Value) -> Value {
    let mut limit = json!({
        "id": id,
        "subject": subject,
        "interval": interval,
        "level": "organization",
        "limit_micros": micros,
        "effective_limit_micros": micros,
        "created_at": "2026-09-24T10:00:00Z",
        "updated_at": "2026-09-24T10:00:00Z",
        "updated_by": "sdk",
    });
    if let (Some(limit), Value::Object(extra)) = (limit.as_object_mut(), extra) {
        limit.extend(extra);
    }
    limit
}

fn page(items: Vec<Value>, next_cursor: Option<&str>) -> Value {
    let mut page = json!({"items": items, "has_more": next_cursor.is_some()});
    if let Some(cursor) = next_cursor {
        page["next_cursor"] = json!(cursor);
    }
    page
}

fn usage(subject: &str) -> Value {
    json!({"subject": subject, "grants": [], "windows": []})
}

fn refused_before_sending(error: Error, says: &str) {
    match error {
        Error::Configuration(message) => assert!(message.contains(says), "{message}"),
        other => panic!("expected a configuration error, got {other:?}"),
    }
}

#[test]
fn an_amount_is_counted_in_micros_exactly() {
    assert_eq!(Usd::dollars(20_000).micros(), 20_000_000_000);
    assert_eq!(Usd::cents(1_250).micros(), 12_500_000);
    assert_eq!(Usd::from_micros(300).micros(), 300);
    assert_eq!(Usd::default().micros(), 0);
    assert!(Usd::dollars(1) > Usd::cents(99));
}

#[test]
fn an_amount_prints_as_dollars_with_cents_and_finer_digits_only_when_it_has_them() {
    assert_eq!(Usd::dollars(20_000).to_string(), "$20000.00");
    assert_eq!(Usd::cents(1_250).to_string(), "$12.50");
    assert_eq!(Usd::from_micros(300).to_string(), "$0.0003");
    assert_eq!(Usd::from_micros(-3_250_000).to_string(), "-$3.25");
    assert_eq!(Usd::default().to_string(), "$0.00");
    assert_eq!(
        Usd::from_micros(i64::MIN).to_string(),
        "-$9223372036854.775808"
    );
}

#[test]
fn an_amount_written_out_is_read_exactly() {
    for (written, micros) in [
        ("12", 12_000_000),
        ("12.5", 12_500_000),
        ("$12.50", 12_500_000),
        (" -3.25 ", -3_250_000),
        ("0.000300", 300),
        (".5", 500_000),
        ("0.1000000", 100_000),
        ("$20000.00", 20_000_000_000),
    ] {
        assert_eq!(
            written.parse::<Usd>().unwrap(),
            Usd::from_micros(micros),
            "{written}"
        );
    }
    for amount in [
        Usd::dollars(20_000),
        Usd::cents(1_250),
        Usd::from_micros(-300),
    ] {
        assert_eq!(amount.to_string().parse::<Usd>().unwrap(), amount);
    }
}

#[test]
fn an_amount_that_is_not_plain_dollars_or_is_finer_than_a_micro_is_refused() {
    for (written, says) in [
        ("", "is not an amount"),
        ("12.", "is not an amount"),
        ("1,000", "is not an amount"),
        ("1e3", "is not an amount"),
        ("twelve", "is not an amount"),
        ("0.1 + 0.2", "is not an amount"),
        ("0.0000001", "finer than a millionth"),
        ("9223372036855", "too large"),
        ("99999999999999999999", "too large"),
    ] {
        refused_before_sending(written.parse::<Usd>().unwrap_err(), says);
    }
}

#[tokio::test]
async fn a_limit_is_created_with_its_amount_in_micros() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/spend/limits",
        201,
        limit(1, "organization", "monthly", 20_000_000_000, json!({})),
    );

    let created = server
        .backend()
        .spend_limits()
        .create(&Limit::new(
            LimitSubject::Organization,
            LimitInterval::Monthly,
            Usd::dollars(20_000),
        ))
        .await
        .unwrap();

    assert_eq!(created.id, 1);
    assert_eq!(
        server.request(Method::POST, "/v1/spend/limits").body,
        json!({"subject": "organization", "interval": "monthly", "limit_micros": 20_000_000_000i64})
    );
}

#[tokio::test]
async fn a_limit_names_the_ids_level_and_note_it_was_given() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/spend/limits",
        201,
        limit(2, "api_key", "daily", 10_000_000, json!({})),
    );
    let limits = server.backend().spend_limits();

    limits
        .create(
            &Limit::new(LimitSubject::ApiKey, LimitInterval::Daily, Usd::dollars(50))
                .app_id("1234"),
        )
        .await
        .unwrap();
    limits
        .create(
            &Limit::new(LimitSubject::ApiKey, LimitInterval::Daily, Usd::dollars(10))
                .app_id("1234")
                .api_key("mz7kq2xa9b")
                .level(LimitLevel::ApiKey)
                .note("CI key"),
        )
        .await
        .unwrap();

    let sent: Vec<Value> = server
        .requests(Method::POST, "/v1/spend/limits")
        .into_iter()
        .map(|seen| seen.body)
        .collect();
    assert_eq!(
        sent,
        vec![
            json!({"subject": "api_key", "interval": "daily", "limit_micros": 50_000_000,
                   "app_id": "1234"}),
            json!({"subject": "api_key", "interval": "daily", "limit_micros": 10_000_000,
                   "app_id": "1234", "api_key": "mz7kq2xa9b", "level": "api_key",
                   "note": "CI key"}),
        ]
    );
}

#[tokio::test]
async fn a_team_or_category_within_an_app_is_named_as_the_router_names_it() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/spend/limits",
        201,
        limit(3, "app_team", "weekly", 1, json!({})),
    );
    let limits = server.backend().spend_limits();

    limits
        .create(
            &Limit::new(LimitSubject::Team, LimitInterval::Weekly, Usd::dollars(1))
                .team("finance")
                .app_id("1234"),
        )
        .await
        .unwrap();
    limits
        .create(
            &Limit::new(
                LimitSubject::Category,
                LimitInterval::Weekly,
                Usd::dollars(1),
            )
            .category("project:onboarding")
            .app_id("1234"),
        )
        .await
        .unwrap();

    let subjects: Vec<Value> = server
        .requests(Method::POST, "/v1/spend/limits")
        .into_iter()
        .map(|seen| seen.body["subject"].clone())
        .collect();
    assert_eq!(subjects, vec![json!("app_team"), json!("app_category")]);
}

#[tokio::test]
async fn a_limit_the_router_would_refuse_is_refused_before_anything_is_sent() {
    let server = Server::start().await;
    let limits = server.backend().spend_limits();
    let daily = |subject| Limit::new(subject, LimitInterval::Daily, Usd::dollars(1));

    for (limit, says) in [
        (daily(LimitSubject::Team), "names its team"),
        (daily(LimitSubject::Category), "names its category"),
        (
            daily(LimitSubject::AppTeam).team("finance"),
            "names its app",
        ),
        (
            daily(LimitSubject::User).team("finance"),
            "team is for the team",
        ),
        (
            daily(LimitSubject::Organization).category("project:x"),
            "category is for the category",
        ),
        (daily(LimitSubject::Team).team("fin ance"), "is not 1 to 59"),
        (
            daily(LimitSubject::Team).team("t".repeat(60)),
            "is not 1 to 59",
        ),
        (
            daily(LimitSubject::Category).category("Team:finance"),
            "names a team",
        ),
        (
            daily(LimitSubject::Category).category("c".repeat(65)),
            "is not 1 to 64",
        ),
        (
            Limit::new(
                LimitSubject::Organization,
                LimitInterval::Daily,
                Usd::from_micros(-1),
            ),
            "below zero",
        ),
    ] {
        refused_before_sending(limits.create(&limit).await.unwrap_err(), says);
    }
    assert!(server.seen().is_empty(), "{:#?}", server.seen());
}

#[tokio::test]
async fn limits_are_listed_page_by_page_with_the_filters_named() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/v1/spend/limits",
        200,
        page(
            vec![limit(
                1,
                "api_key",
                "daily",
                1,
                json!({"api_key": "mz7kq2xa9b"}),
            )],
            Some("next"),
        ),
    );
    server.route(
        Method::GET,
        "/v1/spend/limits",
        200,
        page(vec![limit(2, "api_key", "daily", 2, json!({}))], None),
    );

    let listed = server
        .backend()
        .spend_limits()
        .list(&LimitFilter {
            subject: Some(LimitSubject::ApiKey),
            interval: Some(LimitInterval::Daily),
            app_id: Some("1234".into()),
            api_key: Some("mz7kq2xa9b".into()),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(
        listed.iter().map(|limit| limit.id).collect::<Vec<_>>(),
        [1, 2]
    );
    let queries: Vec<String> = server
        .requests(Method::GET, "/v1/spend/limits")
        .into_iter()
        .map(|seen| seen.query)
        .collect();
    assert_eq!(
        queries,
        [
            "subject=api_key&app_id=1234&api_key_id=mz7kq2xa9b&interval=daily&limit=200",
            "subject=api_key&app_id=1234&api_key_id=mz7kq2xa9b&interval=daily&limit=200&cursor=next",
        ]
    );
}

#[tokio::test]
async fn a_limit_is_read_changed_and_deleted_by_its_id() {
    let server = Server::start().await;
    let one = limit(7, "user", "daily", 25_000_000, json!({}));
    server.route(Method::GET, "/v1/spend/limits/7", 200, one.clone());
    server.route(Method::PATCH, "/v1/spend/limits/7", 200, one);
    server.route(Method::DELETE, "/v1/spend/limits/7", 204, Value::Null);
    let limits = server.backend().spend_limits();

    assert_eq!(limits.get(7).await.unwrap().limit_micros, 25_000_000);
    limits.set_amount(7, Usd::dollars(25)).await.unwrap();
    limits.set_note(7, "").await.unwrap();
    limits.delete(7).await.unwrap();

    let changes: Vec<Value> = server
        .requests(Method::PATCH, "/v1/spend/limits/7")
        .into_iter()
        .map(|seen| seen.body)
        .collect();
    assert_eq!(
        changes,
        vec![json!({"limit_micros": 25_000_000}), json!({"note": ""})]
    );
    server.request(Method::DELETE, "/v1/spend/limits/7");
    refused_before_sending(
        limits
            .set_amount(7, Usd::from_micros(-1))
            .await
            .unwrap_err(),
        "below zero",
    );
}

#[tokio::test]
async fn a_reset_names_the_subject_its_reason_and_the_windows() {
    let server = Server::start().await;
    server.route(
        Method::POST,
        "/v1/spend/limits/usage/reset",
        200,
        usage("user"),
    );
    let limits = server.backend().spend_limits();

    limits
        .reset(
            &Reset::new(LimitSubject::User, "refunded the stuck job")
                .app_id("1234")
                .user_id("user-42")
                .intervals([LimitInterval::Weekly]),
        )
        .await
        .unwrap();
    limits
        .reset(&Reset::new(LimitSubject::Organization, "new quarter"))
        .await
        .unwrap();

    let sent: Vec<Value> = server
        .requests(Method::POST, "/v1/spend/limits/usage/reset")
        .into_iter()
        .map(|seen| seen.body)
        .collect();
    assert_eq!(
        sent,
        vec![
            json!({"subject": "user", "reason": "refunded the stuck job", "intervals": ["weekly"],
                   "app_id": "1234", "user_id": "user-42"}),
            json!({"subject": "organization", "reason": "new quarter"}),
        ]
    );
    refused_before_sending(
        limits
            .reset(&Reset::new(LimitSubject::Organization, "  "))
            .await
            .unwrap_err(),
        "needs a reason",
    );
}

#[tokio::test]
async fn teams_are_listed_within_an_app_when_one_is_named() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/v1/teams",
        200,
        json!([{"name": "finance", "apps": ["1234"], "limits": [], "spent": []}]),
    );
    let teams = server.backend().teams();

    let listed = teams.list(None).await.unwrap();
    teams.list(Some("1234")).await.unwrap();

    assert_eq!(listed[0].name, "finance");
    let queries: Vec<String> = server
        .requests(Method::GET, "/v1/teams")
        .into_iter()
        .map(|seen| seen.query)
        .collect();
    assert_eq!(queries, ["", "app_id=1234"]);
}

#[tokio::test]
async fn a_team_with_no_limit_in_the_interval_gets_one_created() {
    let server = Server::start().await;
    server.route(Method::GET, "/v1/spend/limits", 200, page(vec![], None));
    server.route(
        Method::POST,
        "/v1/spend/limits",
        201,
        limit(
            4,
            "team",
            "monthly",
            8_000_000_000,
            json!({"team": "finance"}),
        ),
    );

    let set = server
        .backend()
        .teams()
        .set_limit(&TeamLimit::new(
            "finance",
            LimitInterval::Monthly,
            Usd::dollars(8_000),
        ))
        .await
        .unwrap();

    assert_eq!(set.id, 4);
    assert_eq!(
        server.request(Method::GET, "/v1/spend/limits").query,
        "subject=team&team=finance&interval=monthly&limit=200"
    );
    assert_eq!(
        server.request(Method::POST, "/v1/spend/limits").body,
        json!({"subject": "team", "interval": "monthly", "limit_micros": 8_000_000_000i64,
               "team": "finance"})
    );
    assert!(
        server
            .requests(Method::PATCH, "/v1/spend/limits/4")
            .is_empty()
    );
}

#[tokio::test]
async fn a_team_with_a_limit_in_the_interval_has_it_changed() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/v1/spend/limits",
        200,
        page(
            vec![
                limit(
                    5,
                    "team",
                    "monthly",
                    1,
                    json!({"team": "finance", "app_id": "1234"}),
                ),
                limit(6, "team", "monthly", 1, json!({"team": "finance"})),
            ],
            None,
        ),
    );
    server.route(
        Method::PATCH,
        "/v1/spend/limits/6",
        200,
        limit(
            6,
            "team",
            "monthly",
            8_000_000_000,
            json!({"team": "finance"}),
        ),
    );

    server
        .backend()
        .teams()
        .set_limit(
            &TeamLimit::new("finance", LimitInterval::Monthly, Usd::dollars(8_000))
                .note("FY27 budget"),
        )
        .await
        .unwrap();

    assert_eq!(
        server.request(Method::PATCH, "/v1/spend/limits/6").body,
        json!({"limit_micros": 8_000_000_000i64, "note": "FY27 budget"})
    );
    assert!(server.requests(Method::POST, "/v1/spend/limits").is_empty());
}

#[tokio::test]
async fn a_team_limit_created_meanwhile_is_changed_instead() {
    let server = Server::start().await;
    let app_team = limit(
        8,
        "app_team",
        "daily",
        1,
        json!({"team": "finance", "app_id": "1234"}),
    );
    server.route(Method::GET, "/v1/spend/limits", 200, page(vec![], None));
    server.route(
        Method::GET,
        "/v1/spend/limits",
        200,
        page(vec![app_team.clone()], None),
    );
    server.route(
        Method::POST,
        "/v1/spend/limits",
        409,
        refusal("conflict", "limit_exists", "the subject has one"),
    );
    server.route(Method::PATCH, "/v1/spend/limits/8", 200, app_team);

    server
        .backend()
        .teams()
        .set_limit(
            &TeamLimit::new("finance", LimitInterval::Daily, Usd::dollars(30)).app_id("1234"),
        )
        .await
        .unwrap();

    assert_eq!(
        server.request(Method::POST, "/v1/spend/limits").body["subject"],
        "app_team"
    );
    assert_eq!(
        server.request(Method::PATCH, "/v1/spend/limits/8").body,
        json!({"limit_micros": 30_000_000})
    );
}

#[tokio::test]
async fn a_team_limit_is_removed_and_one_already_gone_is_no_error() {
    let server = Server::start().await;
    server.route(
        Method::GET,
        "/v1/spend/limits",
        200,
        page(
            vec![limit(9, "team", "weekly", 1, json!({"team": "finance"}))],
            None,
        ),
    );
    server.route(
        Method::DELETE,
        "/v1/spend/limits/9",
        404,
        refusal("not_found", "limit_not_found", "no such limit"),
    );
    let teams = server.backend().teams();

    teams
        .remove_limit("finance", LimitInterval::Weekly, None)
        .await
        .unwrap();
    teams
        .remove_limit("finance", LimitInterval::Monthly, None)
        .await
        .unwrap();

    assert_eq!(
        server.requests(Method::DELETE, "/v1/spend/limits/9").len(),
        1
    );
    refused_before_sending(
        teams
            .remove_limit("team:finance", LimitInterval::Weekly, None)
            .await
            .unwrap_err(),
        "is not 1 to 59",
    );
}
