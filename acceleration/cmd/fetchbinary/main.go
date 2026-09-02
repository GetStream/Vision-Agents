// Command fetchbinary downloads a released router binary from S3 and installs it locally.
//
// It runs as an initContainer beside the router, into an emptyDir the router then execs
// from. The image it ships in is a slow-moving runtime shell: it carries the Opus and soxr
// shared libraries the router links against, and nothing that changes release to release.
// The binary is what changes, and it arrives from here.
//
// Which version to fetch comes from one of two places. -version pins it outright. -version-key
// instead names the registry pointer shiply writes on deploy, resolved here at pod start.
// That is the whole design: Kubernetes owns the image, the version registry owns the binary.
// Putting the version in the pod spec instead would let an unrelated chart deploy roll the
// binary back to whatever the spec last said, while reporting a healthy rollout.
//
// Usage:
//
//	fetchbinary -binary router -version-key MultiRegion/us-east4/Accelerate/current -dest /opt/stream/bin/router
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

const (
	defaultBucket = "stream-puppet"
	defaultPrefix = "releases/r/GetStream/Vision-Agents"
	defaultRegion = "us-east-1"
	defaultMode   = 0o755
	// The version registry shiply writes. It lives in us-east-1 whichever region the pod
	// runs in, so it is a separate client from the one reading the release bucket.
	defaultVersionBucket = "stream-services-version-registry"
	defaultVersionRegion = "us-east-1"
)

func main() {
	bucket := flag.String("bucket", defaultBucket, "S3 bucket holding the releases")
	prefix := flag.String("prefix", defaultPrefix, "key prefix under the bucket, before /<version>/")
	binary := flag.String("binary", "router", "binary to fetch")
	version := flag.String("version", "", "release to pin; takes precedence over -version-key")
	versionKey := flag.String("version-key", "", "registry pointer to resolve the version from")
	versionBucket := flag.String("version-bucket", defaultVersionBucket, "bucket holding the version registry")
	versionRegion := flag.String("version-region", defaultVersionRegion, "region of the version registry bucket")
	arch := flag.String("arch", runtime.GOARCH, "target architecture")
	dest := flag.String("dest", "", "path to install the binary at")
	region := flag.String("region", "", "region for the release bucket; defaults to AWS_REGION then "+defaultRegion)
	mode := flag.Int("mode", defaultMode, "file mode for the installed binary")
	timeout := flag.Duration("timeout", 5*time.Minute, "overall timeout")
	flag.Parse()

	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	err := run(ctx, logger, options{
		Bucket:        *bucket,
		Prefix:        *prefix,
		Binary:        *binary,
		Version:       *version,
		VersionKey:    *versionKey,
		VersionBucket: *versionBucket,
		VersionRegion: *versionRegion,
		Arch:          *arch,
		Dest:          *dest,
		Region:        *region,
		Mode:          os.FileMode(*mode),
	})
	if err != nil {
		logger.Error("could not install the binary", "error", err)
		os.Exit(1)
	}
}

type options struct {
	Bucket        string
	Prefix        string
	Binary        string
	Version       string
	VersionKey    string
	VersionBucket string
	VersionRegion string
	Arch          string
	Dest          string
	Region        string
	Mode          os.FileMode
}

func run(ctx context.Context, logger *slog.Logger, opts options) error {
	if opts.Binary == "" || opts.Dest == "" {
		return errors.New("a binary and a destination are required")
	}
	if opts.Version == "" && opts.VersionKey == "" {
		return errors.New("either a version or a version key is required")
	}

	region := opts.Region
	if region == "" {
		if region = os.Getenv("AWS_REGION"); region == "" {
			region = defaultRegion
		}
	}

	releases, err := client(ctx, region)
	if err != nil {
		return err
	}

	version := opts.Version
	if version == "" {
		registry := releases
		if opts.VersionRegion != region {
			if registry, err = client(ctx, opts.VersionRegion); err != nil {
				return err
			}
		}
		if version, err = resolve(ctx, registry, opts.VersionBucket, opts.VersionKey); err != nil {
			return fmt.Errorf("resolve %s: %w", opts.VersionKey, err)
		}
		logger.Info("resolved the version to install", "version", version, "pointer", opts.VersionKey)
	}

	key := path.Join(opts.Prefix, version, fmt.Sprintf("%s-linux-%s", opts.Binary, opts.Arch))

	want, err := checksum(ctx, releases, opts.Bucket, key+".sha256")
	if err != nil {
		return fmt.Errorf("read the checksum for %s: %w", key, err)
	}

	// A pod that restarts without the emptyDir being cleared already has the binary, and
	// re-downloading it would only delay the restart.
	if have, err := hashOf(opts.Dest); err == nil && have == want {
		logger.Info("installed already, leaving it alone", "version", version, "dest", opts.Dest)
		return nil
	}

	if err := download(ctx, releases, opts.Bucket, key, opts.Dest, want, opts.Mode); err != nil {
		return err
	}

	// The version is recorded beside the binary because the pod spec deliberately does not
	// carry it, so this file is the only place a running pod says what it is running.
	if err := os.WriteFile(opts.Dest+".version", []byte(version), 0o644); err != nil {
		return fmt.Errorf("record the version: %w", err)
	}

	logger.Info("installed", "version", version, "dest", opts.Dest, "sha256", want)
	return nil
}

func client(ctx context.Context, region string) (*s3.Client, error) {
	// Credentials come from the environment: this deployment reads S3 with a scoped static
	// key delivered through Secret Manager rather than by federating an identity.
	cfg, err := awsconfig.LoadDefaultConfig(ctx, awsconfig.WithRegion(region))
	if err != nil {
		return nil, fmt.Errorf("load aws config for %s: %w", region, err)
	}
	return s3.NewFromConfig(cfg), nil
}

// resolve reads the version a registry pointer names.
func resolve(ctx context.Context, api *s3.Client, bucket, key string) (string, error) {
	object, err := api.GetObject(ctx, &s3.GetObjectInput{Bucket: aws.String(bucket), Key: aws.String(key)})
	if err != nil {
		return "", err
	}
	defer object.Body.Close()

	body, err := io.ReadAll(io.LimitReader(object.Body, 256))
	if err != nil {
		return "", err
	}
	version := strings.TrimSpace(string(body))
	if version == "" {
		return "", errors.New("the pointer is empty")
	}
	return version, nil
}

// checksum reads the sidecar published beside the binary. It is fetched before the binary
// rather than after, so a release missing one fails before anything is written.
func checksum(ctx context.Context, api *s3.Client, bucket, key string) (string, error) {
	object, err := api.GetObject(ctx, &s3.GetObjectInput{Bucket: aws.String(bucket), Key: aws.String(key)})
	if err != nil {
		return "", err
	}
	defer object.Body.Close()

	body, err := io.ReadAll(io.LimitReader(object.Body, 256))
	if err != nil {
		return "", err
	}
	return parseChecksum(body)
}

// parseChecksum reads the hash out of a sidecar. One written by sha256sum is
// "<hex>  <filename>", and one written by hand is just the hex.
func parseChecksum(body []byte) (string, error) {
	sum, _, _ := strings.Cut(strings.TrimSpace(string(body)), " ")
	if len(sum) != hex.EncodedLen(sha256.Size) {
		return "", fmt.Errorf("%q is not a sha256", sum)
	}
	if _, err := hex.DecodeString(sum); err != nil {
		return "", fmt.Errorf("%q is not a sha256: %w", sum, err)
	}
	return sum, nil
}

// download streams the binary from S3 into place.
func download(ctx context.Context, api *s3.Client, bucket, key, dest, want string, mode os.FileMode) error {
	object, err := api.GetObject(ctx, &s3.GetObjectInput{Bucket: aws.String(bucket), Key: aws.String(key)})
	if err != nil {
		return fmt.Errorf("get %s: %w", key, err)
	}
	defer object.Body.Close()

	if err := install(object.Body, dest, want, mode); err != nil {
		return fmt.Errorf("install %s: %w", key, err)
	}
	return nil
}

// install writes a binary to a temporary file beside its destination, hashing as it goes,
// and renames it into place only once the hash matches.
//
// The temporary file is in the destination's own directory because a rename across
// filesystems is not atomic, and the point of the rename is that a half-written or
// corrupted file is never at the path the router execs.
func install(body io.Reader, dest, want string, mode os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return fmt.Errorf("make the destination directory: %w", err)
	}

	file, err := os.CreateTemp(filepath.Dir(dest), filepath.Base(dest)+".*")
	if err != nil {
		return fmt.Errorf("create a temporary file: %w", err)
	}
	defer os.Remove(file.Name())

	digest := sha256.New()
	if _, err := io.Copy(io.MultiWriter(file, digest), body); err != nil {
		file.Close()
		return fmt.Errorf("download: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close the temporary file: %w", err)
	}

	if got := hex.EncodeToString(digest.Sum(nil)); got != want {
		return fmt.Errorf("is %s, not the %s its sidecar names", got, want)
	}
	if err := os.Chmod(file.Name(), mode); err != nil {
		return fmt.Errorf("set the file mode: %w", err)
	}
	if err := os.Rename(file.Name(), dest); err != nil {
		return fmt.Errorf("install at %s: %w", dest, err)
	}
	return nil
}

// hashOf is the sha256 of a file already on disk, used to tell whether there is anything
// to do.
func hashOf(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()

	digest := sha256.New()
	if _, err := io.Copy(digest, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}
