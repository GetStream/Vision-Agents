package files

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type FiguresSuite struct {
	suite.Suite
}

func TestFiguresSuite(t *testing.T) {
	suite.Run(t, new(FiguresSuite))
}

const chartPage = "# A bar chart\n\nEvery station is cleaned at the end of each shift.\n\nWeek 1\n\nWeek 2\n\nWeek 3\n\nWeek 4"

func (s *FiguresSuite) TestThePageWrittenOutAgainIsLeftOut() {
	read := "Printed text: “A bar chart”; “Every station is cleaned at the end of each shift.”; “Week 1”; “Week 2”; “Week 3”; “Week 4”.\n\n" +
		"> **Figure:** A bar chart compares Weeks 1–4. Week 4 is the highest."

	s.Equal("> **Figure:** A bar chart compares Weeks 1–4. Week 4 is the highest.", unread(read, chartPage))
}

func (s *FiguresSuite) TestASummaryIsKeptThoughThePageHasEveryWordOfIt() {
	read := "> **Figure:** A bar chart: Week 1, Week 2, Week 3, Week 4."

	s.Equal(read, unread(read, chartPage))
}

func (s *FiguresSuite) TestWordsOnlyAPictureShowsAreKept() {
	read := "Fryer oil changes\n\n> **Figure:** A bar chart titled “Fryer oil changes” shows four bars."

	s.Equal(read, unread(read, "# A photo\n\nEvery station is cleaned at the end of each shift."))
}

func (s *FiguresSuite) TestNothingReadIsNothingKept() {
	s.Empty(unread("", chartPage))
	s.Empty(unread("\n\n", chartPage))
}
