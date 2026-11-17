from vision_agents.core.utils.text import mask_phone_number


class TestMaskPhoneNumber:
    def test_keeps_only_the_last_four_digits(self):
        assert mask_phone_number("+15551234567") == "+*******4567"

    def test_keeps_separators_and_masks_digits(self):
        assert mask_phone_number("(555) 123-4567") == "(***) ***-4567"

    def test_masks_a_short_number_entirely(self):
        assert mask_phone_number("911") == "***"

    def test_missing_number(self):
        assert mask_phone_number(None) == "unknown"
        assert mask_phone_number("") == "unknown"
