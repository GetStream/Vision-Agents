// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:flutter_assistant/smooth_size.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('SmoothSize', () {
    Future<void> show(
      WidgetTester tester,
      double height, {
      bool animates = true,
      bool reduced = false,
    }) => tester.pumpWidget(
      MediaQuery(
        data: MediaQueryData(disableAnimations: reduced),
        child: Directionality(
          textDirection: TextDirection.ltr,
          child: Align(
            alignment: Alignment.topLeft,
            child: SmoothSize(
              duration: const Duration(milliseconds: 400),
              animates: animates,
              child: SizedBox(width: 10, height: height),
            ),
          ),
        ),
      ),
    );

    double height(WidgetTester tester) => tester.getSize(find.byType(SmoothSize)).height;

    testWidgets('follows a child that grows on the next frame when it does not animate', (
      tester,
    ) async {
      await show(tester, 50, animates: false);
      await show(tester, 100, animates: false);
      await tester.pump(const Duration(milliseconds: 16));

      expect(tester.takeException(), isNull);
      expect(height(tester), 100);
    });

    testWidgets('follows a child that grows on the next frame under reduced motion', (
      tester,
    ) async {
      await show(tester, 50, reduced: true);
      await show(tester, 100, reduced: true);
      await tester.pump(const Duration(milliseconds: 16));

      expect(tester.takeException(), isNull);
      expect(height(tester), 100);
    });

    testWidgets('eases into the size of a child that grows', (tester) async {
      await show(tester, 50);
      await show(tester, 100);
      await tester.pump(const Duration(milliseconds: 200));

      expect(height(tester), allOf(greaterThan(50), lessThan(100)));
      await tester.pumpAndSettle();
      expect(height(tester), 100);
    });
  });
}
