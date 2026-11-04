package core

import (
	"strings"
)

// photoKind is captioned with the image kind's one file named by its id (MessageKind.Media),
// and the media lookup that finds it (AI-2740).
const photoKind = captioned + "\n        media:\n          id: $.image.id\n          mime_type: $.image.mime_type" +
	"\n    media_lookup:\n      url: https://graph.example.com/v1/{id}\n      file: $.url\n      host: files.example.com"

// An also kind with media reads its one file, named by its id, and is read for that file even
// without a caption; a file it cannot read is none, and without one and a caption the message
// is still not read, as before kind media.
func (s *ChannelSuite) TestAnAlsoKindsMediaReadsItsFileWithOrWithoutACaption() {
	m, err := ParseManifest(minimal(strings.Replace(baseChannel, "text: $.text", photoKind, 1)))
	s.Require().NoError(err)
	photo := []Media{{ID: "300", MIMEType: "image/jpeg"}}
	for _, tc := range []struct {
		name, message string
		text          string
		media         []Media
		skipped       string
	}{
		{"a photo with a caption", `"type":"image","image":{"caption":"Is this chipped?","id":"300","mime_type":"image/jpeg"}`, "Is this chipped?", photo, ""},
		{"a photo alone", `"type":"image","image":{"id":"300","mime_type":"image/jpeg"}`, "", photo, ""},
		{"a photo with an empty caption", `"type":"image","image":{"caption":"","id":"300","mime_type":"image/jpeg"}`, "", photo, ""},
		{"a caption with no id", `"type":"image","image":{"caption":"x","mime_type":"image/jpeg"}`, "x", nil, ""},
		{"no id and no caption", `"type":"image","image":{"mime_type":"image/jpeg"}`, "", nil, "match $.type"},
		{"an empty id", `"type":"image","image":{"id":"","mime_type":"image/jpeg"}`, "", nil, "match $.type"},
		{"a null id", `"type":"image","image":{"id":null,"mime_type":"image/jpeg"}`, "", nil, "match $.type"},
		{"an id that is an object", `"type":"image","image":{"id":{"v":"300"},"mime_type":"image/jpeg"}`, "", nil, "match $.type"},
		{"no mime type", `"type":"image","image":{"id":"300"}`, "", nil, "match $.type"},
		{"a mime type that is an array", `"type":"image","image":{"id":"300","mime_type":["image/jpeg"]}`, "", nil, "match $.type"},
		{"an id under another kind", `"type":"video","image":{"id":"300","mime_type":"image/jpeg"}`, "", nil, "match $.type"},
		{"a text message, whose image is no file", `"type":"text","text":"hi","image":{"id":"300","mime_type":"image/jpeg"}`, "hi", nil, ""},
	} {
		read, err := m.Channel.Read(m.ID, []byte(`{"chat":"c","from":"u","id":"1",`+tc.message+`}`))
		s.Require().NoError(err, tc.name)
		if tc.skipped != "" {
			s.Empty(read.Messages, tc.name)
			s.Equal([]string{tc.skipped}, read.Skipped, tc.name)
			continue
		}
		s.Require().Len(read.Messages, 1, tc.name)
		s.Equal(tc.text, read.Messages[0].Text, tc.name)
		s.Equal(tc.media, read.Messages[0].Media, tc.name)
	}
}

// A kind's media is the kind's files in place of the block's: a message the kind reads gets
// its one file, not the block's parts; one match reads gets the block's, as before.
func (s *ChannelSuite) TestAnAlsoKindsMediaIsInPlaceOfTheBlocks() {
	block := "\n    media:\n      url: $.parts[*].url\n      mime_type: $.parts[*].mime_type"
	m, err := ParseManifest(minimal(strings.Replace(baseChannel, "text: $.text", photoKind+block, 1)))
	s.Require().NoError(err)
	parts := `"parts":[{"url":"https://cdn.example.com/a.png","mime_type":"image/png"}]`

	read, err := m.Channel.Read(m.ID, []byte(`{"chat":"c","from":"u","id":"1","type":"image","image":{"id":"300","mime_type":"image/jpeg"},`+parts+`}`))
	s.Require().NoError(err)
	s.Require().Len(read.Messages, 1)
	s.Equal([]Media{{ID: "300", MIMEType: "image/jpeg"}}, read.Messages[0].Media)

	read, err = m.Channel.Read(m.ID, []byte(`{"chat":"c","from":"u","id":"1","type":"text","text":"hi",`+parts+`}`))
	s.Require().NoError(err)
	s.Require().Len(read.Messages, 1)
	s.Equal([]Media{{URL: "https://cdn.example.com/a.png", MIMEType: "image/png"}}, read.Messages[0].Media)
}

// The block's media may name its files by id too: each element's id, beside its url-less
// siblings, which are no files.
func (s *ChannelSuite) TestTheBlocksMediaMayNameItsFilesByID() {
	block := "text: $.text\n    media:\n      id: $.parts[*].id\n      mime_type: $.parts[*].mime_type" +
		"\n    media_lookup:\n      url: https://graph.example.com/v1/{id}\n      file: $.url\n      host: files.example.com"
	m, err := ParseManifest(minimal(strings.Replace(baseChannel, "text: $.text", block, 1)))
	s.Require().NoError(err)

	read, err := m.Channel.Read(m.ID, []byte(`{"chat":"c","from":"u","id":"1","parts":[{"id":"a","mime_type":"image/png"},{"url":"https://x.example.com","mime_type":"image/png"}]}`))

	s.Require().NoError(err)
	s.Require().Len(read.Messages, 1)
	s.Equal([]Media{{ID: "a", MIMEType: "image/png"}}, read.Messages[0].Media)
}

func (s *ChannelSuite) TestAnAlsoKindsMediaIsCheckedWhenTheManifestIsParsed() {
	kind := func(media string) string {
		return strings.Replace(photoKind, "\n          id: $.image.id\n          mime_type: $.image.mime_type", media, 1)
	}
	for _, tc := range []struct{ name, also, want string }{
		{"neither url nor id", kind("\n          mime_type: $.image.mime_type"), "channel.messages.also[0].media: names its file by exactly one of url and id"},
		{"both url and id", kind("\n          url: $.image.url\n          id: $.image.id\n          mime_type: $.image.mime_type"),
			"channel.messages.also[0].media: names its file by exactly one of url and id"},
		{"an id that is not a path", kind("\n          id: image.id\n          mime_type: $.image.mime_type"), `channel.messages.also[0].media.id: "image.id" is not a path`},
		{"a url that is not a path", kind("\n          url: image.url\n          mime_type: $.image.mime_type"), `channel.messages.also[0].media.url: "image.url" is not a path`},
		{"no mime type", kind("\n          id: $.image.id"), `channel.messages.also[0].media.mime_type: "" is not a path`},
		{"an id with no lookup", strings.Split(photoKind, "\n    media_lookup:")[0], "channel.messages.also[0].media.id: needs channel.messages.media_lookup"},
		{"an id of every part", kind("\n          id: $.image[*].id\n          mime_type: $.image.mime_type"), `channel.messages.also[0].media.id: "$.image[*].id"`},
	} {
		_, err := ParseManifest(minimal(strings.Replace(baseChannel, "text: $.text", tc.also, 1)))
		s.ErrorContains(err, tc.want, tc.name)
	}
	_, err := ParseManifest(minimal(strings.Replace(baseChannel, "text: $.text", kind("\n          url: $.image.url\n          mime_type: $.image.mime_type"), 1)))
	s.NoError(err, "a kind's file by its url")
	_, err = ParseManifest(minimal(strings.Replace(baseChannel, "text: $.text", photoKind, 1)))
	s.NoError(err)
}

func (s *ChannelSuite) TestAMediaLookupIsCheckedWhenTheManifestIsParsed() {
	lookup := func(url, file, host string) string {
		return strings.Split(photoKind, "\n    media_lookup:")[0] + "\n    media_lookup:\n      url: " + url + "\n      file: " + file + "\n      host: " + host
	}
	const good, file, host = "https://graph.example.com/v1/{id}", "$.url", "files.example.com"
	for _, tc := range []struct{ name, block, want string }{
		{"no url", lookup(`""`, file, host), `channel.messages.media_lookup.url: "" names no {id}`},
		{"a url with no {id}", lookup("https://graph.example.com/v1/media", file, host), "names no {id}, so it would find every file at one URL"},
		{"{id} in the host", lookup("https://{id}.example.com/v1/{id}", file, host), "{id} is in the host"},
		{"http", lookup("http://graph.example.com/v1/{id}", file, host), "does not start with https://"},
		{"an undeclared name", lookup("https://graph.example.com/{version}/{id}", file, host), "{version} is not a declared input"},
		{"no file", lookup(good, `""`, host), `channel.messages.media_lookup.file: "" is not a path`},
		{"a file of every element", lookup(good, "$.data[*].url", host), "channel.messages.media_lookup.file: a lookup answers one file, so it has no [*]"},
		{"no host", lookup(good, file, `""`), `channel.messages.media_lookup.host: "" is not a lowercase host name alone`},
		{"a host with a scheme", lookup(good, file, "https://files.example.com"), "channel.messages.media_lookup.host"},
		{"a host with a port", lookup(good, file, "files.example.com:443"), "channel.messages.media_lookup.host"},
		{"a host with a path", lookup(good, file, "files.example.com/x"), "channel.messages.media_lookup.host"},
		{"a host with userinfo", lookup(good, file, "u@files.example.com"), "channel.messages.media_lookup.host"},
		{"an uppercase host", lookup(good, file, "Files.example.com"), "channel.messages.media_lookup.host"},
	} {
		_, err := ParseManifest(minimal(strings.Replace(baseChannel, "text: $.text", tc.block, 1)))
		s.ErrorContains(err, tc.want, tc.name)
	}
	_, err := ParseManifest(minimal(strings.Replace(baseChannel, "text: $.text", lookup(good, file, host), 1)))
	s.NoError(err)
}

// The id goes into the lookup's URL as a value from a message goes into a reply's: one that
// would leave its path segment is refused, so the credential only goes where the URL says.
func (s *ChannelSuite) TestAMediaLookupTakesTheIDAsOnePathSegment() {
	m, err := ParseManifest(minimal(strings.Replace(baseChannel, "text: $.text", photoKind, 1)))
	s.Require().NoError(err)
	resolved, err := m.Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)

	target, err := resolved.MediaLookupURL("1003383421387256")
	s.Require().NoError(err)
	s.Equal("https://graph.example.com/v1/1003383421387256", target)
	for _, id := range []string{"", "..", ".", "1/../../evil", "1?x=y", "1#x", "%2F", "a b"} {
		_, err := resolved.MediaLookupURL(id)
		s.Error(err, "%q", id)
	}

	plain, err := ParseManifest(minimal(baseChannel))
	s.Require().NoError(err)
	without, err := plain.Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	_, err = without.MediaLookupURL("1")
	s.ErrorContains(err, "has no channel.messages.media_lookup")
}

// The connection's credential goes with the GET of the file a lookup answers, so the file is
// requested only at an https URL on the lookup's host, with no userinfo and no port.
func (s *ChannelSuite) TestAMediaLookupsFileIsOnlyOnItsHost() {
	m, err := ParseManifest(minimal(strings.Replace(baseChannel, "text: $.text", photoKind, 1)))
	s.Require().NoError(err)
	resolved, err := m.Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	for _, tc := range []struct{ name, answer, want string }{
		{"the host", `{"url":"https://files.example.com/a/?mid=1&hash=x","id":"1"}`, "https://files.example.com/a/?mid=1&hash=x"},
		{"the host in another case", `{"url":"https://FILES.example.com/a"}`, "https://FILES.example.com/a"},
		{"http", `{"url":"http://files.example.com/a"}`, ""},
		{"another host", `{"url":"https://evil.example.com/a"}`, ""},
		{"a host under it", `{"url":"https://x.files.example.com/a"}`, ""},
		{"a host it starts", `{"url":"https://files.example.com.evil.example/a"}`, ""},
		{"its host with a trailing dot", `{"url":"https://files.example.com./a"}`, ""},
		{"userinfo", `{"url":"https://evil.example@files.example.com/a"}`, ""},
		{"a port", `{"url":"https://files.example.com:8443/a"}`, ""},
		{"port 443", `{"url":"https://files.example.com:443/a"}`, ""},
		{"no scheme", `{"url":"//files.example.com/a"}`, ""},
		{"a relative URL", `{"url":"/a"}`, ""},
		{"no url", `{"id":"1"}`, ""},
		{"a url that is an object", `{"url":{"href":"https://files.example.com/a"}}`, ""},
		{"a url that is a number", `{"url":1}`, ""},
		{"an empty url", `{"url":""}`, ""},
		{"an array", `[{"url":"https://files.example.com/a"}]`, ""},
		{"null", `null`, ""},
		{"not JSON", `<html>`, ""},
	} {
		file, err := resolved.MediaFile([]byte(tc.answer))
		if tc.want == "" {
			s.Error(err, tc.name)
			continue
		}
		s.Require().NoError(err, tc.name)
		s.Equal(tc.want, file, tc.name)
	}
}
