package tools

import (
	"errors"
	"fmt"
)

// UserError is a failure the model is told as written: "there is no order 1042", "the card
// was declined". Of any other failure it is told only that the tool failed, because what a
// database or an API says when it fails can name hosts, accounts and queries; that is logged
// where the function runs instead. Wrapped, its message still reaches the model, without the
// words around it.
type UserError string

func (e UserError) Error() string { return string(e) }

// refused is the failure of a call its function's Authorize turned down. The model is told
// what a UserError says, and otherwise only that the call was refused: why is the caller's
// to log, not the model's to read.
func refused(name string, err error) error {
	var said UserError
	if errors.As(err, &said) {
		return err
	}
	return fmt.Errorf("%w: %w", UserError(name+" was refused"), err)
}
