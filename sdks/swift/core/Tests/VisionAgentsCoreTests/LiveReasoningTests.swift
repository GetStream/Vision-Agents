import Testing

@testable import VisionAgentsCore

/// The windows of thinking the router puts on a reply's live updates, as `reasoningWindow` in
/// `acceleration/internal/conversation/reasoning.go` writes them.
@Suite struct LiveReasoningTests {
    @Test func windowsInOrderMakeTheWholeThinking() {
        var reasoning = LiveReasoning()

        reasoning.add("The order was ", at: 0, to: "r1", of: "reply")
        reasoning.add("delivered unopened.", at: 14, to: "r1", of: "reply")

        #expect(reasoning[message: "reply", step: "r1"] == "The order was delivered unopened.")
    }

    @Test func aRepeatOfRecentThinkingAddsOnlyWhatIsNew() {
        var reasoning = LiveReasoning()
        reasoning.add("Checking the policy", at: 0, to: "r1", of: "reply")

        // A keyframe repeats the last of what was sent, and runs on past it.
        reasoning.add("the policy, which allows 30 days", at: 9, to: "r1", of: "reply")
        reasoning.add("policy", at: 13, to: "r1", of: "reply")

        #expect(reasoning[message: "reply", step: "r1"] == "Checking the policy, which allows 30 days")
    }

    @Test func aMissedWindowLeavesAParagraphBreakRatherThanRunningOn() {
        var reasoning = LiveReasoning()
        reasoning.add("First thought.", at: 0, to: "r1", of: "reply")

        reasoning.add("Much later.", at: 40, to: "r1", of: "reply")

        #expect(reasoning[message: "reply", step: "r1"] == "First thought.\n\nMuch later.")
    }

    @Test func somebodyJoiningMidwayStartsFromTheWindowTheyGet() {
        var reasoning = LiveReasoning()

        reasoning.add("the rest of it", at: 300, to: "r2", of: "reply")

        #expect(reasoning[message: "reply", step: "r2"] == "the rest of it")
        #expect(reasoning[message: "reply", step: "r1"] == nil)
    }

    @Test func positionsCountUnicodeScalarsAsTheRouterDoes() {
        var reasoning = LiveReasoning()
        reasoning.add("café ", at: 0, to: "r1", of: "reply")

        reasoning.add(" 🌸 bloom", at: 4, to: "r1", of: "reply")

        #expect(reasoning[message: "reply", step: "r1"] == "café 🌸 bloom")
    }

    /// Every reply numbers its steps from `r1`, so a later reply's first round is not the
    /// earlier one's, whether it thinks for less or for more.
    @Test func eachReplyKeepsTheThinkingOfItsOwnSteps() {
        var reasoning = LiveReasoning()
        reasoning.add("Looking up the order history and the return policy.", at: 0, to: "r1", of: "first")

        reasoning.add("Refund it.", at: 0, to: "r1", of: "second")
        reasoning.add("Checking which of the two orders arrived damaged, then the refund window.", at: 0, to: "r1", of: "third")

        #expect(reasoning[message: "first", step: "r1"] == "Looking up the order history and the return policy.")
        #expect(reasoning[message: "second", step: "r1"] == "Refund it.")
        #expect(reasoning[message: "third", step: "r1"] == "Checking which of the two orders arrived damaged, then the refund window.")
    }
}
