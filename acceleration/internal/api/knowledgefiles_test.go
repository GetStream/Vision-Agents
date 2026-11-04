//go:build integration

package api

import (
	"bytes"
	"context"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// KnowledgeFilesSuite covers files uploaded to a knowledge base: read in the background into
// passages that say their page, scans read from their pictures, and the files that cannot
// be read refused with why.
type KnowledgeFilesSuite struct {
	RouterSuite
	namespace string
}

func TestKnowledgeFilesSuite(t *testing.T) {
	runSuite(t, new(KnowledgeFilesSuite))
}

func (s *KnowledgeFilesSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.namespace = "files-" + uuid.NewString()
}

func (s *KnowledgeFilesSuite) TestAPDFIsReadIntoPassagesThatSayTheirPage() {
	status, sent, _ := s.upload("manual.pdf", extractFixture(s.T(), "manual.pdf"), "")
	s.Require().Equal(http.StatusAccepted, status)
	s.Equal(KnowledgeDocumentStatePending, sent.State)
	s.Equal("application/pdf", sent.ContentType)

	read := s.settled(sent.Id)

	s.Equal(KnowledgeDocumentStateIndexed, read.State)
	s.Nil(read.Error)
	s.Equal(12, read.Pages)
	s.Positive(read.Passages)
	s.Contains(*read.Text, "<!-- page 7 -->")
	passage := s.passageContaining("Fryer oil is changed every 48 hours")
	s.True(strings.HasPrefix(passage, "manual.pdf > p. 7"), passage)
	stored, err := s.store.KnowledgeDocument(context.Background(), s.customerID(), sent.Id)
	s.Require().NoError(err)
	s.Empty(stored.ObjectKey, "the file is deleted once read")
}

func (s *KnowledgeFilesSuite) TestAScannedPageIsReadFromItsPictureAndBilledAsOCR() {
	since := time.Now().UTC()
	_, sent, _ := s.upload("scanned.pdf", extractFixture(s.T(), "scanned.pdf"), "")

	read := s.settled(sent.Id)

	s.Equal(KnowledgeDocumentStateIndexed, read.State)
	s.Contains(*read.Text, "This first page has a text layer.")
	s.Contains(*read.Text, "<!-- page 2 -->\n\n# Fryer cleaning\n\nDrain the oil.", "the scan is read by the OCR router")
	rows := s.meteredRows(routing.OCR, since, 1)
	s.Equal("pages", rows[0].Provider)
}

func (s *KnowledgeFilesSuite) TestAFileWithAPasswordFailsSayingSo() {
	_, sent, _ := s.upload("encrypted.pdf", extractFixture(s.T(), "encrypted.pdf"), "")

	read := s.settled(sent.Id)

	s.Equal(KnowledgeDocumentStateFailed, read.State)
	s.Require().NotNil(read.Error)
	s.Contains(*read.Error, "protected with a password")
	s.Zero(read.Passages)
}

func (s *KnowledgeFilesSuite) TestADamagedFileFailsSayingSo() {
	_, sent, _ := s.upload("corrupt.pdf", extractFixture(s.T(), "corrupt.pdf"), "")

	read := s.settled(sent.Id)

	s.Equal(KnowledgeDocumentStateFailed, read.State)
	s.Contains(*read.Error, "damaged")
}

func (s *KnowledgeFilesSuite) TestAPhotoIsRefusedWhileItsSenderWaits() {
	photo := append([]byte{0xFF, 0xD8, 0xFF, 0xE0, 0, 0x10}, []byte("JFIF\x00 a picture")...)

	status, _, answer := s.upload("menu.jpg", photo, "")

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(answer), "Upload a PDF")
	s.Empty(s.listed(), "nothing is kept of a file that cannot be read")
}

func (s *KnowledgeFilesSuite) TestAFileOverFiftyMegabytesIsRefused() {
	status, _, answer := s.upload("huge.txt", bytes.Repeat([]byte("a"), 50<<20+1), "")

	s.Equal(http.StatusRequestEntityTooLarge, status, string(answer))
	s.Empty(s.listed())
}

func (s *KnowledgeFilesSuite) TestUploadingTheSameSourceAgainReplacesTheDocument() {
	_, first, _ := s.upload("handbook.docx", extractFixture(s.T(), "handbook.docx"), "handbook")
	s.settled(first.Id)

	_, second, _ := s.upload("manual.pdf", extractFixture(s.T(), "manual.pdf"), "handbook")
	read := s.settled(second.Id)

	s.Equal(first.Id, second.Id, "a source is one document")
	s.Equal(12, read.Pages)
	s.Len(s.listed(), 1)
}

func (s *KnowledgeFilesSuite) TestADocumentRemovedBeforeItsFileIsReadLeavesNothing() {
	_, sent, _ := s.upload("removed.pdf", extractFixture(s.T(), "manual.pdf"), "")

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/knowledge/documents/"+sent.Id, nil, nil))

	// Removed before the read or after it, the end is the same: no passage of it is found.
	s.Never(func() bool {
		_, passages := s.knowledge.stored()
		for _, passage := range passages {
			if strings.HasPrefix(passage.Source, "removed.pdf") {
				return true
			}
		}
		return false
	}, time.Second, 20*time.Millisecond)
	s.Empty(s.listed())
}

func (s *KnowledgeFilesSuite) TestTextPostedIsIndexedAtOnce() {
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/knowledge", map[string]any{
		"namespace": s.namespace,
		"documents": []map[string]string{{"source": "hours.md", "text": "# Hours\n\nNine to five, every day but Sunday."}},
	}, nil))

	listed := s.listed()

	s.Require().Len(listed, 1)
	s.Equal(KnowledgeDocumentStateIndexed, listed[0].State)
	s.Equal("text/markdown", listed[0].ContentType)
	s.EqualValues(len("# Hours\n\nNine to five, every day but Sunday."), listed[0].Bytes)
}

func (s *KnowledgeFilesSuite) TestAFormWithoutAFileIsRefused() {
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	s.Require().NoError(form.WriteField("note", "no file here"))
	s.Require().NoError(form.Close())

	status, answer := s.send(s.serverClient, body.Bytes(), form.FormDataContentType(), url.Values{"namespace": {s.namespace}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(answer), "no part named file")
}

func (s *KnowledgeFilesSuite) TestANamespaceIsRequired() {
	body, contentType := formWith("notes.md", []byte("# Notes"))

	status, _ := s.send(s.serverClient, body, contentType, url.Values{})

	s.Equal(http.StatusBadRequest, status)
}

func (s *KnowledgeFilesSuite) TestOnlyTheCustomersOwnBackendMayUpload() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		body, contentType := formWith("notes.md", []byte("# Notes"))
		status, _ := s.send(as, body, contentType, url.Values{"namespace": {s.namespace}})
		return status
	})
}

// upload sends a file as the customer's backend, under source when it is given.
func (s *KnowledgeFilesSuite) upload(name string, data []byte, source string) (int, IndexedKnowledgeDocument, []byte) {
	body, contentType := formWith(name, data)
	query := url.Values{"namespace": {s.namespace}}
	if source != "" {
		query.Set("source", source)
	}
	status, answer := s.send(s.serverClient, body, contentType, query)
	var sent IndexedKnowledgeDocument
	if status == http.StatusAccepted {
		s.Require().NoError(json.Unmarshal(answer, &sent), string(answer))
	}
	return status, sent, answer
}

func (s *KnowledgeFilesSuite) send(as *testClient, body []byte, contentType string, query url.Values) (int, []byte) {
	request, err := http.NewRequest(http.MethodPost, as.base()+knowledgeFilesPath+"?"+query.Encode(), bytes.NewReader(body))
	s.Require().NoError(err)
	request.Header = as.header.Clone()
	request.Header.Set("Content-Type", contentType)
	response, err := s.server.Client().Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	answer, err := readAll(response)
	s.Require().NoError(err)
	return response.StatusCode, answer
}

// settled is the document once its file has been read, or has failed to be.
func (s *KnowledgeFilesSuite) settled(id string) IndexedKnowledgeDocument {
	var read IndexedKnowledgeDocument
	s.Require().Eventually(func() bool {
		read = IndexedKnowledgeDocument{}
		status := s.serverClient.do(http.MethodGet, "/v1/agents/knowledge/documents/"+id, nil, &read)
		return status == http.StatusOK && read.State != KnowledgeDocumentStatePending
	}, 30*time.Second, 20*time.Millisecond, "the file was never read")
	return read
}

func (s *KnowledgeFilesSuite) listed() []IndexedKnowledgeDocument {
	var listed []IndexedKnowledgeDocument
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/knowledge/documents?namespace="+s.namespace, nil, &listed))
	return listed
}

// passageContaining is the source of the stored passage that holds text.
func (s *KnowledgeFilesSuite) passageContaining(text string) string {
	_, passages := s.knowledge.stored()
	for _, passage := range passages {
		if strings.Contains(passage.Text, text) {
			return passage.Source
		}
	}
	s.FailNow("no passage holds the text", text)
	return ""
}

// formWith is a multipart form holding one file, as a browser's FormData sends it.
func formWith(name string, data []byte) ([]byte, string) {
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	part, err := form.CreateFormFile("file", name)
	if err != nil {
		panic(err)
	}
	if _, err := part.Write(data); err != nil {
		panic(err)
	}
	if err := form.Close(); err != nil {
		panic(err)
	}
	return body.Bytes(), form.FormDataContentType()
}

// extractFixture is one of the files extract is tested against.
func extractFixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "knowledge", "extract", "testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return data
}
