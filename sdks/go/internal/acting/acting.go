// Package acting names the end user a backend speaks for, to the packages of this module.
//
// stream.Backend.ActingFor is deprecated for a per-call UserID, and a deprecated field used
// from another package is flagged by staticcheck. The packages of this module that still act
// for a user, one request at a time, hand it to Backend.Acting as a Who instead. Nothing
// outside this module can import this package, so Acting is no public way to do it.
package acting

// Who is the end user a server-side credential speaks for. The zero value names nobody.
type Who struct{ id string }

// User is the end user with this id. An empty id names nobody.
func User(id string) Who { return Who{id: id} }

// ID is the user's id, or empty for nobody.
func (w Who) ID() string { return w.id }
