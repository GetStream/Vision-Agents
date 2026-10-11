import Testing

@testable import VisionAgentsCore

/// The windows of thinking the router puts on a reply's live updates, as `reasoningWindow` in
/// `acceleration/internal/conversation/reasoning.go` writes them.
@Suite struct LiveReasoningTests {
    @Test func windowsInOrderMakeTheWholeThinking() {
        var reasoning = LiveReasoning()

        reasoning.add("The order was ", at: 0, to: "r1")
        reasoning.add("delivered unopened.", at: 14, to: "r1")

        #expect(reasoning["r1"] == "The order was delivered unopened.")
    }

    @Test func aRepeatOfRecentThinkingAddsOnlyWhatIsNew() {
        var reasoning = LiveReasoning()
        reasoning.add("Checking the policy", at: 0, to: "r1")

        // A keyframe repeats the last of what was sent, and runs on past it.
        reasoning.add("the policy, which allows 30 days", at: 9, to: "r1")
        reasoning.add("policy", at: 13, to: "r1")

        #expect(reasoning["r1"] == "Checking the policy, which allows 30 days")
    }

    @Test func aMissedWindowLeavesAParagraphBreakRatherThanRunningOn() {
        var reasoning = LiveReasoning()
        reasoning.add("First thought.", at: 0, to: "r1")

        reasoning.add("Much later.", at: 40, to: "r1")

        #expect(reasoning["r1"] == "First thought.\n\nMuch later.")
    }

    @Test func somebodyJoiningMidwayStartsFromTheWindowTheyGet() {
        var reasoning = LiveReasoning()

        reasoning.add("the rest of it", at: 300, to: "r2")

        #expect(reasoning["r2"] == "the rest of it")
        #expect(reasoning["r1"] == nil)
    }

    @Test func positionsCountUnicodeScalarsAsTheRouterDoes() {
        var reasoning = LiveReasoning()
        reasoning.add("café ", at: 0, to: "r1")

        reasoning.add(" 🌸 bloom", at: 4, to: "r1")

        #expect(reasoning["r1"] == "café 🌸 bloom")
    }
}
