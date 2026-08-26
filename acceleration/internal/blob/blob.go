// Package blob stores the files a customer uploads, in whichever object store the
// deployment runs against.
//
// One URL selects the store, so S3 and GCS are the same code path and a test or a laptop
// can use a directory. Credentials come from the ambient provider chain rather than from
// options, matching how every other secret in this service is already an environment
// variable.
package blob

import (
	"context"
	"fmt"
	"io"

	gcblob "gocloud.dev/blob"
	_ "gocloud.dev/blob/fileblob"
	_ "gocloud.dev/blob/gcsblob"
	_ "gocloud.dev/blob/s3blob"
	"gocloud.dev/gcerrors"
)

// EnvURL is the environment variable naming the bucket, for example
// s3://voices?region=eu-west-1, gs://voices, or file:///var/lib/router/voices.
const EnvURL = "ROUTER_BLOB_URL"

// Bucket is somewhere to put files.
type Bucket struct {
	bucket *gcblob.Bucket
}

// Open opens the bucket a URL names. An empty URL is a deployment that stores no files,
// which is a nil bucket rather than an error: the paths that need one say so themselves.
func Open(ctx context.Context, url string) (*Bucket, error) {
	if url == "" {
		return nil, nil
	}

	bucket, err := gcblob.OpenBucket(ctx, url)
	if err != nil {
		return nil, fmt.Errorf("blob: open %s: %w", url, err)
	}
	return &Bucket{bucket: bucket}, nil
}

// Write stores a file and returns how many bytes it held.
func (b *Bucket) Write(ctx context.Context, key, contentType string, content io.Reader) (int64, error) {
	writer, err := b.bucket.NewWriter(ctx, key, &gcblob.WriterOptions{ContentType: contentType})
	if err != nil {
		return 0, fmt.Errorf("blob: write %s: %w", key, err)
	}

	written, err := io.Copy(writer, content)
	if err != nil {
		// The failed write is closed and then removed, so a half-written object is not
		// left behind for something later to read as if it were whole.
		writer.Close()
		if err := b.bucket.Delete(ctx, key); err != nil {
			return 0, fmt.Errorf("blob: write %s: %w", key, err)
		}
		return 0, fmt.Errorf("blob: write %s: %w", key, err)
	}
	if err := writer.Close(); err != nil {
		return 0, fmt.Errorf("blob: write %s: %w", key, err)
	}
	return written, nil
}

// Read returns a whole file. Callers store things small enough to hold, so there is no
// streaming read.
func (b *Bucket) Read(ctx context.Context, key string) ([]byte, error) {
	content, err := b.bucket.ReadAll(ctx, key)
	if err != nil {
		return nil, fmt.Errorf("blob: read %s: %w", key, err)
	}
	return content, nil
}

// Delete removes a file. One that is already gone is not an error, because the caller
// wanted it gone.
func (b *Bucket) Delete(ctx context.Context, key string) error {
	err := b.bucket.Delete(ctx, key)
	if err == nil || gcerrors.Code(err) == gcerrors.NotFound {
		return nil
	}
	return fmt.Errorf("blob: delete %s: %w", key, err)
}

// Close releases the bucket.
func (b *Bucket) Close() error { return b.bucket.Close() }
