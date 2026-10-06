// Package stack records where an error was first seen, which a Go error does not carry.
//
// Wrap an error where it enters this code base -- returned by a library, the standard
// library or a vendor's SDK, or made here with errors.New or fmt.Errorf -- and pass it up
// unchanged from there. The stack is that of the first Wrap; wrapping again is a no-op, so
// a function never has to know whether its callee already did.
package stack

import (
	"errors"
	"fmt"
	"io"
	"runtime"
	"strconv"
	"strings"
)

// maxDepth bounds the frames kept. A handler's error is rarely more than a few dozen frames
// below net/http, and everything under that is net/http itself.
const maxDepth = 64

type withStack struct {
	err error
	pcs []uintptr
}

// Wrap returns err with the stack of its caller attached, or err itself when something in
// its chain already carries one. Wrap(nil) is nil.
func Wrap(err error) error {
	if err == nil {
		return nil
	}
	if _, ok := errors.AsType[*withStack](err); ok {
		return err
	}
	pcs := make([]uintptr, maxDepth)
	// Skip runtime.Callers and Wrap, so the first frame is the one that called Wrap.
	n := runtime.Callers(2, pcs)
	return &withStack{err: err, pcs: pcs[:n]}
}

func (w *withStack) Error() string { return w.err.Error() }

func (w *withStack) Unwrap() error { return w.err }

// Format prints the error alone for %s and %v, and the error followed by its stack for %+v.
func (w *withStack) Format(f fmt.State, verb rune) {
	switch {
	case verb == 'v' && f.Flag('+'):
		_, _ = io.WriteString(f, w.err.Error())
		_, _ = io.WriteString(f, "\n")
		_, _ = io.WriteString(f, w.trace())
	case verb == 'q':
		_, _ = io.WriteString(f, strconv.Quote(w.err.Error()))
	default:
		_, _ = io.WriteString(f, w.err.Error())
	}
}

// trace renders the stack one frame per two lines, function then file:line, the shape
// runtime/debug.Stack uses.
func (w *withStack) trace() string {
	var b strings.Builder
	frames := runtime.CallersFrames(w.pcs)
	for {
		frame, more := frames.Next()
		b.WriteString(frame.Function)
		b.WriteString("\n\t")
		b.WriteString(frame.File)
		b.WriteString(":")
		b.WriteString(strconv.Itoa(frame.Line))
		b.WriteString("\n")
		if !more {
			break
		}
	}
	return b.String()
}

// Trace returns the stack recorded by the Wrap in err's chain, or "" when there is none.
func Trace(err error) string {
	if w, ok := errors.AsType[*withStack](err); ok {
		return w.trace()
	}
	return ""
}
