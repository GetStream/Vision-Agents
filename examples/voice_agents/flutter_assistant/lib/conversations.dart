import 'dart:async';

import 'package:flutter/material.dart';
import 'package:stream_core_flutter/core.dart';
import 'package:intl/intl.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

import 'assistant.dart';

/// This person's conversations with the assistant, most recently active first.
///
/// The router keeps them: a session belongs to whoever opened it, so this lists this phone's
/// user's and nobody else's. It titles each one itself once there is something to go on, and
/// a conversation that ended carries on when it is next asked something, so there is nothing
/// to keep here but the list.
class Conversations extends ChangeNotifier {
  /// The conversations found, newest first.
  List<Session> items = const [];

  /// What the list is narrowed to, or empty for all of them.
  String searchText = '';

  /// Why the list could not be read, or null.
  String? failure;

  /// How many reads there have been, so only the latest one's answer is shown: a slow search
  /// for what was typed earlier must not replace the one for what is typed now.
  var _reads = 0;

  /// Reads the list again, narrowed to [searchText] when there is one.
  Future<void> reload() async {
    _reads += 1;
    final read = _reads;
    try {
      final page = searchText.isEmpty
          ? await Assistant.sessions.query(limit: 50)
          : await Assistant.sessions.search(searchText, limit: 50);
      if (read != _reads) return;
      // Only the ones kept in writing can be opened again as a chat: a session that was only
      // ever a call, or incognito, has no channel to read back.
      items = [
        for (final session in page.items)
          if (session.conversationId.isNotEmpty) session,
      ];
      failure = null;
    } on Object catch (error) {
      if (read != _reads) return;
      failure = '$error';
    }
    notifyListeners();
  }

  /// Deletes a conversation: its turns, and what it remembered, go with it.
  Future<void> delete(Session session) async {
    // A read already on its way began before the delete and would bring the row back.
    _reads += 1;
    items = [
      for (final item in items)
        if (item.id != session.id) item,
    ];
    notifyListeners();
    try {
      await Assistant.sessions.delete(session.id);
    } on Object catch (error) {
      // The list read back puts the row where it was, and then says why it is still there.
      await reload();
      failure = 'Could not delete "${session.displayTitle}": $error';
      notifyListeners();
    }
  }

  /// The conversations under Today, Yesterday, This week and then their month, in the
  /// router's order.
  ///
  /// A day can appear twice: the list is in order of activity, and the days are when each
  /// conversation started, which is all a [Session] here says.
  static List<({String title, List<Session> items})> sections(
    List<Session> sessions, {
    required DateTime now,
  }) {
    final days = <({String title, List<Session> items})>[];
    for (final session in sessions) {
      final title = dayTitle(session.createdAt, now: now);
      if (days.lastOrNull?.title == title) {
        days.last.items.add(session);
      } else {
        days.add((title: title, items: [session]));
      }
    }
    return days;
  }

  /// What a conversation started at [date] is listed under, seen at [now].
  ///
  /// Counted in the phone's calendar days, so a conversation is under This week for the five
  /// days before yesterday, whatever the time it started.
  static String dayTitle(DateTime date, {required DateTime now}) {
    final local = date.toLocal();
    // The days as UTC dates, so a night when the clocks change is a day like any other.
    final today = DateTime.utc(now.year, now.month, now.day);
    final day = DateTime.utc(local.year, local.month, local.day);
    final daysAgo = today.difference(day).inDays;
    if (daysAgo <= 0) return 'Today';
    if (daysAgo == 1) return 'Yesterday';
    if (daysAgo < 7) return 'This week';
    return DateFormat('MMMM y').format(local);
  }
}

extension SessionTitle on Session {
  /// What a conversation is called in the list: the router's title once it has given it one.
  String get displayTitle => title.isEmpty ? 'New conversation' : title;
}

/// The sidebar: a new conversation, a search, and the conversations there were, by day.
class ConversationList extends StatefulWidget {
  const ConversationList({
    super.key,
    required this.conversations,
    required this.selected,
    required this.onNew,
    required this.onSettings,
    required this.onOpen,
    required this.onDelete,
  });

  final Conversations conversations;

  /// The conversation open beside this, or null for a new one.
  final String? selected;
  final VoidCallback onNew;
  final VoidCallback onSettings;
  final ValueChanged<Session> onOpen;
  final ValueChanged<Session> onDelete;

  @override
  State<ConversationList> createState() => _ConversationListState();
}

class _ConversationListState extends State<ConversationList> {
  late final _search = TextEditingController(text: widget.conversations.searchText);
  Timer? _searching;

  @override
  void dispose() {
    _searching?.cancel();
    _search.dispose();
    super.dispose();
  }

  /// Waits for typing to pause before searching.
  void _searchChanged(String text) {
    widget.conversations.searchText = text;
    _searching?.cancel();
    _searching = Timer(
      text.isEmpty ? Duration.zero : const Duration(milliseconds: 300),
      widget.conversations.reload,
    );
  }

  @override
  Widget build(BuildContext context) {
    final colors = context.streamColorScheme;
    final text = context.streamTextTheme;
    final spacing = context.streamSpacing;
    final conversations = widget.conversations;
    return SafeArea(
      child: ListenableBuilder(
        listenable: conversations,
        builder: (context, _) {
          final sections = Conversations.sections(conversations.items, now: DateTime.now());
          return Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Row(
                children: [
                  Expanded(
                    child: StreamListTile(
                      leading: Icon(context.streamIcons.edit),
                      title: const Text('New chat'),
                      titleTextStyle: text.bodyEmphasis.copyWith(color: colors.textPrimary),
                      onTap: widget.onNew,
                    ),
                  ),
                  Padding(
                    padding: EdgeInsets.only(right: spacing.xs),
                    child: StreamButton.icon(
                      style: .secondary,
                      type: .ghost,
                      tooltip: 'Settings',
                      onPressed: widget.onSettings,
                      icon: const Icon(Icons.settings_outlined),
                    ),
                  ),
                ],
              ),
              Padding(
                padding: EdgeInsets.symmetric(horizontal: spacing.md),
                child: StreamTextInput(
                  controller: _search,
                  onChanged: _searchChanged,
                  textInputAction: TextInputAction.search,
                  leading: Icon(context.streamIcons.search),
                  hintText: 'Search conversations',
                ),
              ),
              Expanded(
                child: RefreshIndicator(
                  onRefresh: conversations.reload,
                  child: ListView(
                    children: [
                      for (final section in sections) ...[
                        Padding(
                          padding: EdgeInsets.fromLTRB(
                            spacing.md,
                            spacing.md,
                            spacing.md,
                            spacing.xxs,
                          ),
                          child: Text(
                            section.title,
                            style: text.captionEmphasis.copyWith(color: colors.textSecondary),
                          ),
                        ),
                        for (final session in section.items)
                          _ConversationRow(
                            session: session,
                            isSelected: session.id == widget.selected,
                            onOpen: () => widget.onOpen(session),
                            onDelete: () => widget.onDelete(session),
                          ),
                      ],
                      if (conversations.failure case final failure?)
                        Padding(
                          padding: EdgeInsets.all(spacing.md),
                          child: Text(
                            failure,
                            style: text.captionDefault.copyWith(color: colors.accentError),
                          ),
                        ),
                    ],
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

/// A conversation in the list, which a swipe deletes.
class _ConversationRow extends StatelessWidget {
  const _ConversationRow({
    required this.session,
    required this.isSelected,
    required this.onOpen,
    required this.onDelete,
  });

  final Session session;
  final bool isSelected;
  final VoidCallback onOpen;
  final VoidCallback onDelete;

  @override
  Widget build(BuildContext context) {
    final colors = context.streamColorScheme;
    final text = context.streamTextTheme;
    return Dismissible(
      key: ValueKey(session.id),
      direction: DismissDirection.endToStart,
      onDismissed: (_) => onDelete(),
      background: Container(
        color: colors.accentError,
        alignment: Alignment.centerRight,
        padding: EdgeInsets.symmetric(horizontal: context.streamSpacing.md),
        child: Text('Delete', style: text.bodyEmphasis.copyWith(color: colors.textOnAccent)),
      ),
      child: StreamListTile(
        title: Text(session.displayTitle, maxLines: 1, overflow: TextOverflow.ellipsis),
        titleTextStyle: (isSelected ? text.bodyEmphasis : text.bodyDefault).copyWith(
          color: colors.textPrimary,
        ),
        selected: isSelected,
        onTap: onOpen,
      ),
    );
  }
}
