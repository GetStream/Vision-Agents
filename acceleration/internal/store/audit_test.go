//go:build integration

package store

func (s *StoreSuite) TestAPriceListChangeIsRecordedWithNoCustomer() {
	entry := AuditEntry{
		ResourceType: AuditResourcePriceList, ResourceID: "change-" + newID(), Action: AuditCreated,
		Source: AuditSourceManage, ActorID: "staff:" + newID(), ActorName: "Łukasz",
		Changes: []AuditChange{{Field: "unit_price_micros", After: 1}},
	}

	s.Require().NoError(s.store.RecordAuditEntry(s.ctx, &entry))

	var kept AuditEntry
	s.Require().NoError(s.store.DB().NewSelect().Model(&kept).Where("id = ?", entry.ID).Scan(s.ctx))
	s.Empty(kept.CustomerID)
	s.Equal(AuditSourceManage, kept.Source)
	s.Equal("Łukasz", kept.ActorName)
}

func (s *StoreSuite) TestAPriceListChangeNamingACustomerIsRefused() {
	entry := AuditEntry{
		CustomerID: "acme-app", ResourceType: AuditResourcePriceList, ResourceID: "change-" + newID(),
		Action: AuditCreated, Changes: []AuditChange{{Field: "unit_price_micros", After: 1}},
	}

	s.Error(s.store.RecordAuditEntry(s.ctx, &entry))
}

func (s *StoreSuite) TestAnAppsChangeWithNoCustomerIsRefused() {
	entry := AuditEntry{
		ResourceType: AuditPolicy, ResourceID: newID(), Action: AuditCreated,
		Changes: []AuditChange{{Field: "budget", After: 1}},
	}

	s.Error(s.store.RecordAuditEntry(s.ctx, &entry))
}

func (s *StoreSuite) TestNoClientCanNameItselfTheManagementAPI() {
	s.Equal(AuditSourceAPI, AuditSource(AuditSourceManage))
	s.Equal(AuditSourceCLI, AuditSource(AuditSourceCLI))
}
