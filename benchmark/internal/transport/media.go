package transport

// Frame is one 20 ms chunk from the far end.
type Frame struct {
	PCM []int16
}

// Media is a bidirectional PCM pipe at audio.Rate.
type Media interface {
	Send(pcm []int16) error
	Recv() <-chan Frame
	Close() error
}
