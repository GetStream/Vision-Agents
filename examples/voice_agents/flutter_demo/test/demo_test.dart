import 'package:flutter_demo/demo.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

void main() {
  test('lookup_order finds an order whatever the case of its number', () async {
    final output = await _tool('lookup_order').run({'order_id': 'a-1042'});

    expect(output.text, startsWith('Order A-1042: 2 Larkspur wool throws'));
  });

  test('lookup_order says when there is no such order', () async {
    final output = await _tool('lookup_order').run({'order_id': 'B-7'});

    expect(output.text, 'There is no order B-7 on this account.');
  });

  test('issue_refund refunds the amount asked for to the card', () async {
    final output = await _tool(
      'issue_refund',
    ).run({'order_id': 'a-1042', 'amount': '78.00', 'reason': 'unopened'});

    expect(output.text, startsWith('Refunded 78.00 for order A-1042'));
  });

  test(
    'issue_refund refunds nothing for an order that is not on the account',
    () async {
      final output = await _tool(
        'issue_refund',
      ).run({'order_id': 'B-7', 'amount': '78.00', 'reason': 'unopened'});

      expect(output.text, 'There is no order B-7 on this account.');
    },
  );

  test('issue_refund asks the person first', () {
    expect(_tool('issue_refund').approval?.title, 'Refund this order?');
    expect(_tool('lookup_order').approval, isNull);
  });
}

AgentTool _tool(String name) =>
    Demo.tools.singleWhere((tool) => tool.name == name);
