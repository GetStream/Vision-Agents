import 'convert.dart';
import 'generated/api.dart' as api;
import 'models.dart';

/// The user's scheduled tasks: what agents said they would do for them later.
///
/// A device sees and changes only its own user's. Tasks are made by the agent in a
/// conversation, or by the app's backend; a device lists, changes, pauses, resumes and cancels
/// them.
final class ScheduledTasks {
  ScheduledTasks(this._operations);

  final api.Operations _operations;

  /// One page of this user's tasks, soonest first.
  Future<ListPage<ScheduledTask>> list([
    ScheduledTaskQuery query = const ScheduledTaskQuery(),
  ]) async {
    final page = await _operations.listScheduledTasks(
      limit: query.limit,
      cursor: _named(query.cursor),
      state: query.state?.name,
      configId: _named(query.configId),
      sessionId: _named(query.sessionId),
    );
    return ListPage(
      [for (final task in page.items) scheduledTaskOf(task)],
      hasMore: page.hasMore,
      nextCursor: _named(page.nextCursor),
    );
  }

  /// One task. Somebody else's is reported as not found.
  Future<ScheduledTask> get(String id) async =>
      scheduledTaskOf(await _operations.getScheduledTask(id: id));

  /// Changes a task and returns it as it now is. Null leaves a field as it is.
  ///
  /// [schedule] replaces the whole schedule, but for a timezone it leaves out. [next] moves only
  /// the next occurrence: a snooze, or a skip when it lands past the one after. Either lands 15
  /// minutes from now at the soonest, unless it is no earlier than the current next occurrence.
  Future<ScheduledTask> update(
    String id, {
    String? title,
    ScheduledTaskSchedule? schedule,
    ScheduledTaskNext? next,
    List<String>? deliverTo,
    ScheduledTaskQuietHours? quietHours,
  }) async => scheduledTaskOf(
    await _operations.updateScheduledTask(
      id: id,
      body: scheduledTaskChangeOf(
        title: title,
        schedule: schedule,
        next: next,
        deliverTo: deliverTo,
        quietHours: quietHours,
      ),
    ),
  );

  /// Stops a task coming due until it is resumed.
  Future<ScheduledTask> pause(String id) async =>
      scheduledTaskOf(await _operations.pauseScheduledTask(id: id));

  /// Starts a paused or failed task again, counting its next occurrence from now. A one-shot
  /// whose time came while it was paused is completed instead, and a task with no occurrence
  /// left needs a new schedule first.
  Future<ScheduledTask> resume(String id) async =>
      scheduledTaskOf(await _operations.resumeScheduledTask(id: id));

  /// Cancels a task: it and its runs are gone.
  Future<void> delete(String id) => _operations.deleteScheduledTask(id: id);

  /// One page of the times a task came due, newest first. Snoozes and pauses write none. Pass
  /// the page's [ListPage.nextCursor] as [cursor] for the next one.
  Future<ListPage<ScheduledTaskRun>> runs(String id, {int? limit, String? cursor}) async {
    final page = await _operations.listScheduledTaskRuns(
      id: id,
      limit: limit,
      cursor: _named(cursor),
    );
    return ListPage(
      [for (final run in page.items) scheduledTaskRunOf(run)],
      hasMore: page.hasMore,
      nextCursor: _named(page.nextCursor),
    );
  }
}

/// A name left empty, which means the same as leaving it out.
String? _named(String? name) => name == null || name.isEmpty ? null : name;
