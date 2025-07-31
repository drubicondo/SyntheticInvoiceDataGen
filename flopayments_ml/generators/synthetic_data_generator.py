import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import random
import uuid
from typing import Dict, List, Tuple, Optional
from faker import Faker
import logging

from ..core.data_models import Fattura, Transazione
from ..core.data_types import MatchType, QualityLevel, TimingPattern, AmountPattern, GroundTruth
from dataclasses import asdict
from ..core.exceptions import ValidationError
from .ai_text_generator import AITextGenerator
from ..utils.file_utils import check_write_permission
from ..utils.export_utils import csv_to_xlsx_sheets

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    def __init__(self, config: Dict, azure_endpoint: str):
        self.config = config
        self.fake = Faker('it_IT')
        self.batch_size = self.config.get('batch_size', 10)
        self.ai_generator = AITextGenerator(azure_endpoint)
        self.companies = []
        self.company_invoice_history = {}  # Track invoice history per company
        self.recurring_patterns = {}       # Track recurring patterns
        self.sectors = [
            "Consulenza IT", "Servizi Legali", "Marketing", "Contabilità", 
            "Ingegneria", "Architettura", "Formazione", "Logistica"
        ]
        self.service_types = {
            "Consulenza IT": ["Sviluppo software", "Manutenzione sistemi", "Consulenza tecnica"],
            "Servizi Legali": ["Consulenza legale", "Assistenza contrattuale", "Rappresentanza"],
            "Marketing": ["Campagne pubblicitarie", "Social media management", "Branding"],
            "Contabilità": ["Tenuta contabilità", "Consulenza fiscale", "Bilanci"],
            "Ingegneria": ["Progettazione", "Collaudi", "Consulenza tecnica"],
            "Architettura": ["Progettazione architettonica", "Direzione lavori", "Pratiche edilizie"],
            "Formazione": ["Corsi di formazione", "Seminari", "Coaching"],
            "Logistica": ["Trasporti", "Magazzinaggio", "Distribuzione"]
        }
        
        # Validate configuration
        self._validate_config()
        
        # Generate companies and initialize history
        num_companies = self.config.get('num_companies', 20)
        self.companies = self.generate_companies(num_companies)
        self._initialize_company_histories()
        self._setup_recurrency_patterns()
    
    def _initialize_company_histories(self):
        """Initialize invoice history for all companies"""
        for company in self.companies:
            company_id = company['id']
            if company_id not in self.company_invoice_history:
                self.company_invoice_history[company_id] = []
    
    def _validate_config(self):
        """Validate that the configuration is consistent"""
        scenarios = self.config.get('scenarios', {})
        num_companies = self.config.get('num_companies', 20)
        
        # Calculate total invoices needed
        total_invoices = (
            scenarios.get('perfect_1_1', 0) +
            scenarios.get('installments_1_n', 0) +
            scenarios.get('group_payment_n_1', 0) * 3 +  # Assume avg 3 invoices per group
            scenarios.get('standalone_invoices', 0)
        )
        
        # Check if we have enough companies
        min_companies_needed = max(1, total_invoices // 10)  # At least 1 company per 10 invoices
        
        if num_companies < min_companies_needed:
            raise ValueError(
                f"Number of companies ({num_companies}) is too low for the number of invoices to generate ({total_invoices}). "
                f"Minimum recommended: {min_companies_needed} companies."
            )
        
        if num_companies > total_invoices:
            logger.warning(
                f"Number of companies ({num_companies}) is higher than total invoices ({total_invoices}). "
                "Some companies may not have any invoices."
            )
    
    def _setup_recurrency_patterns(self):
        """Setup recurrency patterns for companies"""
        recurrency_config = self.config.get('recurrency_patterns', {})
        
        for company in self.companies:
            company_id = company['id']
            self.company_invoice_history[company_id] = []
            
            # Determine recurrency patterns for this company
            patterns = {
                'has_recurring_clients': random.random() < recurrency_config.get('recurring_clients', 0.3),
                'provides_similar_services': random.random() < recurrency_config.get('similar_services', 0.4),
                'has_monthly_services': random.random() < recurrency_config.get('monthly_services', 0.2),
                'has_project_based': random.random() < recurrency_config.get('project_based', 0.3)
            }
            
            # Define preferred service types for consistency
            sector_services = self.service_types[company['settore']]
            patterns['preferred_services'] = random.sample(
                sector_services, 
                min(2, len(sector_services))  # 1-2 preferred services
            )
            
            # Define recurring client names for this company
            if patterns['has_recurring_clients']:
                patterns['recurring_clients'] = [
                    self._generate_client_name(company['settore']) 
                    for _ in range(random.randint(2, 5))
                ]
            
            self.recurring_patterns[company_id] = patterns
    
    def _generate_client_name(self, sector: str) -> str:
        """Generate a client name appropriate for the sector"""
        base_name = self.fake.company()
        
        # Add sector-appropriate suffixes
        if sector in ["Consulenza IT", "Ingegneria"]:
            suffixes = ["Tech", "Systems", "Solutions", "Digital"]
        elif sector in ["Servizi Legali", "Contabilità"]:
            suffixes = ["& Partners", "Associati", "Studio"]
        elif sector == "Marketing":
            suffixes = ["Media", "Creative", "Brand", "Communications"]
        else:
            suffixes = ["Group", "SPA", "SRL"]
        
        if random.random() < 0.3:  # 30% chance to add suffix
            base_name += " " + random.choice(suffixes)
        
        return base_name
    
    def generate_companies(self, n: int) -> List[Dict]:
        """Generate realistic company data with enhanced sector distribution"""
        companies = []
        
        # Ensure good distribution across sectors
        sector_counts = {sector: 0 for sector in self.sectors}
        
        for i in range(n):
            # Choose sector with some balancing
            if i < len(self.sectors):
                # First round: one company per sector
                sector = self.sectors[i]
            else:
                # Subsequent rounds: weighted random selection
                min_count = min(sector_counts.values())
                available_sectors = [s for s, count in sector_counts.items() if count == min_count]
                sector = random.choice(available_sectors)
            
            sector_counts[sector] += 1
            
            company = {
                'id': str(uuid.uuid4()),
                'nome': self._generate_company_name(sector),
                'piva': self.fake.vat_id(),
                'settore': sector,
                'iban': self.fake.iban()
            }
            companies.append(company)
        
        return companies
    
    def _generate_company_name(self, sector: str) -> str:
        """Generate company name appropriate for the sector"""
        base_name = self.fake.company()
        
        # Add sector-specific elements
        if sector == "Consulenza IT":
            if random.random() < 0.4:
                tech_words = ["Tech", "Digital", "Systems", "Solutions", "Software"]
                base_name = base_name.replace("SRL", "").replace("SPA", "").strip()
                base_name += " " + random.choice(tech_words)
        elif sector == "Servizi Legali":
            if random.random() < 0.3:
                base_name = "Studio Legale " + base_name.replace("SRL", "").replace("SPA", "").strip()
        elif sector == "Marketing":
            if random.random() < 0.3:
                marketing_words = ["Creative", "Media", "Brand", "Communications"]
                base_name += " " + random.choice(marketing_words)
        
        return base_name
    
    def generate_invoice(self, company: Dict, scenario_type: str, amount_range: Optional[Tuple[float, float]] = None, use_recurrency: bool = False) -> Fattura:
        """Generate a single invoice for a company."""
        company_id = company['id']
        patterns = self.recurring_patterns.get(company_id, {})
        
        # Generate emission date
        data_emissione = self.fake.date_between(start_date='-1y', end_date='today')
        
        # Generate amount
        if amount_range:
            importo = random.uniform(*amount_range)
        else:
            importo = random.uniform(100, 5000)  # Default amount range
            if scenario_type == "installment":
                importo = np.random.lognormal(mean=8.5, sigma=0.8)  # €3000-€15000
            else:
                importo = np.random.lognormal(mean=7.5, sigma=1.0)  # €500-€5000
        
        # Determine data_scadenza based on rules and amount
        if random.random() < 0.75: # 75% invoices have data_scadenza = data_emissione
            data_scadenza = data_emissione
        else: # 25% invoices have varied data_scadenza
            # Define possible day ranges and their weights based on amount
            if importo < 1000: # Smaller amounts tend to have shorter due dates
                choices = [(1, 15), (28, 31), (60, 62)]
                weights = [0.6, 0.3, 0.1]
            elif importo < 5000: # Medium amounts
                choices = [(1, 15), (28, 31), (60, 62)]
                weights = [0.2, 0.5, 0.3]
            else: # Larger amounts tend to have longer due dates
                choices = [(1, 15), (28, 31), (60, 62)]
                weights = [0.1, 0.3, 0.6]

            min_days, max_days = random.choices(choices, weights=weights, k=1)[0]
            data_scadenza = data_emissione + timedelta(days=random.randint(min_days, max_days))

        # Choose service type with recurrency consideration
        if use_recurrency and patterns.get('provides_similar_services', False):
            # Use preferred services for consistency
            tipo_servizio = random.choice(patterns.get('preferred_services', 
                                                      self.service_types[company['settore']]))
        else:
            tipo_servizio = random.choice(self.service_types[company['settore']])
        
        # Generate committente with recurrency patterns
        if use_recurrency and patterns.get('has_recurring_clients', False) and random.random() < 0.6:
            # 60% chance to use recurring client
            committente = random.choice(patterns.get('recurring_clients', [self._generate_client_name(company['settore'])]))
        else:
            committente = self._generate_client_name(company['settore'])
        
        # Generate description and numero_fattura using AI
        descrizione, ai_committente, numero_fattura = self.ai_generator.generate_invoice_data(
            company['id'], data_emissione.strftime('%Y-%m-%d'), company['settore'], company['nome'], importo, tipo_servizio  # Fixed: use 'id' instead of 'cliente_id'
        )
        
        # Use AI-generated committente if recurrency doesn't apply
        if not (use_recurrency and patterns.get('has_recurring_clients', False)):
            committente = ai_committente
        
        fattura = Fattura(
            id=uuid.uuid4(),  # Use a unique ID for each invoice
            data_emissione=data_emissione,
            data_scadenza=data_scadenza,
            numero_fattura=numero_fattura,
            descrizione=descrizione,
            importo=importo,
            prestatore=company['nome'],
            committente=committente
        )
        
        # Track invoice history
        self.company_invoice_history[company_id].append({
            'fattura_id': fattura.id,
            'committente': committente,
            'tipo_servizio': tipo_servizio,
            'importo': importo,
            'data_emissione': data_emissione
        })
        
        return fattura

    def _chunk_list(self, items: List, size: int) -> List[List]:
        """Split a list into chunks of a given size."""
        return [items[i:i + size] for i in range(0, len(items), size)]

    def _generate_invoices_batch(self, companies: List[Dict], scenario_type: str, amount_range: Optional[Tuple[float, float]] = None, use_recurrency: bool = False) -> List[Fattura]:
        """Generate invoices for a batch of companies using a single AI call."""
        prepared = []
        ai_inputs = []

        for company in companies:
            company_id = company['id']
            patterns = self.recurring_patterns.get(company_id, {})

            data_emissione = self.fake.date_between(start_date='-1y', end_date='today')
            data_scadenza = data_emissione + timedelta(days=random.randint(30, 90))

            if amount_range:
                importo = random.uniform(*amount_range)
            else:
                importo = random.uniform(100, 5000)
                if scenario_type == "installment":
                    importo = np.random.lognormal(mean=8.5, sigma=0.8)
                else:
                    importo = np.random.lognormal(mean=7.5, sigma=1.0)

            if use_recurrency and patterns.get('provides_similar_services', False):
                tipo_servizio = random.choice(patterns.get('preferred_services', self.service_types[company['settore']]))
            else:
                tipo_servizio = random.choice(self.service_types[company['settore']])

            if use_recurrency and patterns.get('has_recurring_clients', False) and random.random() < 0.6:
                committente = random.choice(patterns.get('recurring_clients', [self._generate_client_name(company['settore'])]))
            else:
                committente = self._generate_client_name(company['settore'])

            ai_inputs.append({
                'company_id': company['id'],
                'data_emissione': data_emissione.strftime('%Y-%m-%d'),
                'settore': company['settore'],
                'prestatore': company['nome'],
                'importo': importo,
                'tipo_servizio': tipo_servizio
            })

            prepared.append({
                'company': company,
                'company_id': company_id,
                'data_emissione': data_emissione,
                'data_scadenza': data_scadenza,
                'importo': importo,
                'tipo_servizio': tipo_servizio,
                'committente': committente,
                'patterns': patterns,
                'use_recurrency': use_recurrency
            })

        ai_results = self.ai_generator.generate_invoice_data_batch(ai_inputs)
        fatture = []
        for prep, (descrizione, ai_committente, numero_fattura) in zip(prepared, ai_results):
            committente = prep['committente']
            if not (prep['use_recurrency'] and prep['patterns'].get('has_recurring_clients', False)):
                committente = ai_committente

            fattura = Fattura(
                id=uuid.uuid4(),
                data_emissione=prep['data_emissione'],
                data_scadenza=prep['data_scadenza'],
                numero_fattura=numero_fattura,
                descrizione=descrizione,
                importo=prep['importo'],
                prestatore=prep['company']['nome'],
                committente=committente
            )

            self.company_invoice_history[prep['company_id']].append({
                'fattura_id': fattura.id,
                'committente': committente,
                'tipo_servizio': prep['tipo_servizio'],
                'importo': prep['importo'],
                'data_emissione': prep['data_emissione']
            })

            fatture.append(fattura)

        return fatture

    def _generate_payments_batch(self, fatture: list[Fattura], 
                                 amount_patterns: list[AmountPattern], 
                                 timing_patterns: list[TimingPattern],
                                 quality_levels: list[QualityLevel]) -> list[Transazione]:
        """Generate payments for a batch of invoices."""
        prepared = []
        ai_inputs = []

        for fattura, company, amount_pattern, timing_pattern, quality_level in zip(fatture, companies, amount_patterns, timing_patterns, quality_levels):
            base_amount = fattura.importo
            if amount_pattern == AmountPattern.EXACT:
                importo = base_amount
            elif amount_pattern == AmountPattern.PARTIAL:
                importo = base_amount * random.uniform(0.3, 0.8)
            elif amount_pattern == AmountPattern.EXCESS:
                importo = base_amount * random.uniform(1.01, 1.1)
            elif amount_pattern == AmountPattern.DISCOUNT:
                importo = base_amount * random.uniform(0.9, 0.98)
            elif amount_pattern == AmountPattern.PENALTY:
                importo = base_amount * random.uniform(1.02, 1.05)
            else:
                importo = base_amount

            if timing_pattern == TimingPattern.STANDARD:
                days_offset = random.randint(0, 90)
                data_pagamento = fattura.data_emissione + timedelta(days=days_offset)
            elif timing_pattern == TimingPattern.DELAYED:
                days_offset = random.randint(91, 180)
                data_pagamento = fattura.data_emissione + timedelta(days=days_offset)
            elif timing_pattern == TimingPattern.EARLY:
                days_offset = random.randint(-30, -1)
                data_pagamento = fattura.data_emissione + timedelta(days=days_offset)
            elif timing_pattern == TimingPattern.SAME_DAY:
                data_pagamento = fattura.data_emissione
            else:
                days_offset = random.randint(0, 90)
                data_pagamento = fattura.data_emissione + timedelta(days=days_offset)

            if quality_level == QualityLevel.PERFECT:
                invoice_prob = 0.5
            elif quality_level == QualityLevel.FUZZY:
                invoice_prob = 0.25
            else:
                invoice_prob = 0.1

            include_invoice = random.random() < invoice_prob

            ai_inputs.append({
                'fattura': fattura,
                'importo': importo,
                'include_invoice_number': include_invoice
            })

            prepared.append({
                'data_pagamento': data_pagamento,
                'importo': importo,
                'include_invoice': include_invoice
            })

        ai_results = self.ai_generator.generate_transaction_data_batch(ai_inputs)

        transazioni = []
        for prep, (dettaglio, causale, controparte, has_invoice_ref, is_fallback) in zip(prepared, ai_results):
            transazione = Transazione(
                id=str(uuid.uuid4()),
                data=prep['data_pagamento'],
                dettaglio=dettaglio,
                importo=prep['importo'],
                tipologia_movimento="pagamento",
                controparte=controparte,
                causale=causale,
                invoice_number=1 if has_invoice_ref else 0,
                is_fallback=is_fallback # Add this line
            )
            transazioni.append(transazione)

        return transazioni

    def generate_payment(self, fattura: Fattura, company: Dict, 
                        amount_pattern: AmountPattern = AmountPattern.EXACT,
                        timing_pattern: TimingPattern = TimingPattern.STANDARD,
                        quality_level: QualityLevel = QualityLevel.NOISY) -> Transazione:
        """Generate payment transaction for a given invoice"""
        
        # Calculate payment amount based on pattern
        base_amount = fattura.importo
        if amount_pattern == AmountPattern.EXACT:
            importo = base_amount
        elif amount_pattern == AmountPattern.PARTIAL:
            importo = base_amount * random.uniform(0.3, 0.8)
        elif amount_pattern == AmountPattern.EXCESS:
            importo = base_amount * random.uniform(1.01, 1.1)
        elif amount_pattern == AmountPattern.DISCOUNT:
            importo = base_amount * random.uniform(0.9, 0.98)
        elif amount_pattern == AmountPattern.PENALTY:
            importo = base_amount * random.uniform(1.02, 1.05)
        else:
            importo = base_amount
        
        # Calculate payment date based on timing pattern
        if timing_pattern == TimingPattern.STANDARD:
            # 0-90 days after invoice date
            days_offset = random.randint(0, 90)
            data_pagamento = fattura.data_emissione + timedelta(days=days_offset)
        elif timing_pattern == TimingPattern.DELAYED:
            # >90 days after invoice date
            days_offset = random.randint(91, 180)
            data_pagamento = fattura.data_emissione + timedelta(days=days_offset)
        elif timing_pattern == TimingPattern.EARLY:
            # Before invoice date (advance payment)
            days_offset = random.randint(-30, -1)
            data_pagamento = fattura.data_emissione + timedelta(days=days_offset)
        elif timing_pattern == TimingPattern.SAME_DAY:
            # Same day as invoice
            data_pagamento = fattura.data_emissione
        else:
            # Default to standard
            days_offset = random.randint(0, 90)
            data_pagamento = fattura.data_emissione + timedelta(days=days_offset)
        
        # Determine invoice number inclusion probability based on quality level
        if quality_level == QualityLevel.PERFECT:
            invoice_number_probability = 0.5
        elif quality_level == QualityLevel.FUZZY:
            invoice_number_probability = 0.25
        else:  # NOISY
            invoice_number_probability = 0.1
        
        # Generate transaction details using AI
        dettaglio, causale, controparte, has_invoice_ref = self.ai_generator.generate_transaction_data(
            fattura, importo, invoice_number_probability
        )
        
        transazione = Transazione(
            id=str(uuid.uuid4()),
            data=data_pagamento,
            dettaglio=dettaglio,
            importo=importo,
            tipologia_movimento="pagamento",
            controparte=controparte,
            causale=causale,
            invoice_number=1 if has_invoice_ref else 0
        )
        
        return transazione
    
    def generate_scenario_1_1_perfect(self, n_pairs: int) -> Tuple[List[Fattura], List[Transazione], List[GroundTruth]]:
        """Generate 1:1 perfect match scenario"""
        fatture = []
        transazioni = []
        ground_truth = []

        companies_to_use = self._select_companies_for_scenario(n_pairs)

        for batch in self._chunk_list(companies_to_use, self.batch_size):
            batch_fatture = self._generate_invoices_batch(batch, scenario_type="oneshot", use_recurrency=True)
            fatture.extend(batch_fatture)

            batch_trans = self._generate_payments_batch(
                batch_fatture,
                batch,
                [AmountPattern.EXACT] * len(batch_fatture),
                [TimingPattern.STANDARD] * len(batch_fatture),
                [QualityLevel.NOISY] * len(batch_fatture)
            )
            transazioni.extend(batch_trans)

            for f, t in zip(batch_fatture, batch_trans):
                gt = GroundTruth(
                    fattura_id=str(f.id),
                    pagamento_id=str(t.id),
                    match_type=MatchType.EXACT.value,
                    confidence=1.0,
                    amount_covered=f.importo,
                    notes="Perfect 1:1 match"
                )
                ground_truth.append(gt)

        return fatture, transazioni, ground_truth
    
    def _select_companies_for_scenario(self, n_items: int) -> List[Dict]:
        """Select companies for a scenario, ensuring good distribution and reuse"""
        if n_items <= len(self.companies):
            return random.sample(self.companies, n_items)
        else:
            # Need to reuse companies
            selected = []
            companies_pool = self.companies.copy()
            
            while len(selected) < n_items:
                if not companies_pool:
                    companies_pool = self.companies.copy()
                
                company = random.choice(companies_pool)
                selected.append(company)
                companies_pool.remove(company)
            
            return selected
    
    def _generate_scadenza_date(self, data_emissione: datetime) -> datetime:
        """Generate a due date 30-90 days after the emission date."""
        return data_emissione + timedelta(days=random.randint(30, 90))

    def _generate_group_invoice_amount(self) -> float:
        """Generate invoice amount for group payment scenario (€200-€2000)"""
        return random.uniform(200, 2000)
    
    def _generate_billing_period_invoices(self, company: Dict, n_invoices: int, 
                                         billing_period: str = "monthly") -> List[Fattura]:
        """Generate multiple invoices within the same billing period"""
        fatture = []
        
        # Define a random reference "today" within the last month so that
        # generated billing periods are not always anchored to the real current
        # date. This keeps the dataset temporally consistent but still varied
        # over time.
        real_today = datetime.today().date()
        today_range_start = real_today - relativedelta(months=1)
        reference_today = self.fake.date_between_dates(today_range_start, real_today)

        if billing_period == "monthly":
            start_range_start = reference_today - relativedelta(months=6)
            start_range_end = reference_today - relativedelta(months=1)
            period_start = self.fake.date_between_dates(start_range_start, start_range_end)
            # Last day of the invoice month
            period_end = (period_start.replace(day=1) + relativedelta(months=1)) - timedelta(days=1)
        elif billing_period == "quarterly":
            start_range_start = reference_today - relativedelta(months=9)
            start_range_end = reference_today - relativedelta(months=3)
            period_start = self.fake.date_between_dates(start_range_start, start_range_end)
            period_end = period_start + relativedelta(months=3) - timedelta(days=1)
        else:  # weekly
            start_range_start = reference_today - relativedelta(months=3)
            start_range_end = reference_today - relativedelta(weeks=2)
            period_start = self.fake.date_between_dates(start_range_start, start_range_end)
            period_end = period_start + timedelta(days=7) - timedelta(days=1)
        
        # Generate project/service linking patterns
        link_type = random.choice(["project_code", "monthly_service", "recurring_delivery", "none"])
        
        if link_type == "project_code":
            project_code = f"PROJ-{random.randint(1000, 9999)}"
            base_description = f"Servizi progetto {project_code}"
        elif link_type == "monthly_service":
            service_name = random.choice(["Consulenza", "Manutenzione", "Assistenza", "Formazione"])
            base_description = f"{service_name} mensile"
        elif link_type == "recurring_delivery":
            delivery_type = random.choice(["Consegna", "Fornitura", "Spedizione"])
            base_description = f"{delivery_type} periodica"
        else:
            base_description = None
        
        invoice_inputs = []
        meta = []
        for i in range(n_invoices):
            data_emissione = self.fake.date_between(start_date=period_start, end_date=period_end)
            data_scadenza = self._generate_scadenza_date(data_emissione)
            importo = self._generate_group_invoice_amount()

            if base_description:
                if link_type == "project_code":
                    descrizione_suffix = f" - Fase {i+1}"
                elif link_type == "monthly_service":
                    descrizione_suffix = f" - Periodo {data_emissione.strftime('%m/%Y')}"
                elif link_type == "recurring_delivery":
                    descrizione_suffix = f" - N. {i+1}/{n_invoices}"
                else:
                    descrizione_suffix = ""
                tipo_servizio = f"{base_description}{descrizione_suffix}"
            else:
                tipo_servizio = random.choice(self.service_types[company['settore']])

            invoice_inputs.append({
                'company_id': company['id'],
                'data_emissione': data_emissione.strftime('%Y-%m-%d'),
                'settore': company['settore'],
                'prestatore': company['nome'],
                'importo': importo,
                'tipo_servizio': tipo_servizio
            })
            meta.append((data_emissione, data_scadenza, importo))

        results = self.ai_generator.generate_invoice_data_batch(invoice_inputs)
        for (data_emissione, data_scadenza, importo), (descrizione, committente, numero_fattura) in zip(meta, results):
            fattura = Fattura(
                id=uuid.uuid4(),
                data_emissione=data_emissione,
                data_scadenza=data_scadenza,
                numero_fattura=numero_fattura,
                descrizione=descrizione,
                importo=importo,
                prestatore=company['nome'],
                committente=committente
            )
            fatture.append(fattura)
        
        return fatture
    
    def _generate_group_payment_date(self, fatture: List[Fattura]) -> datetime:
        """Generate payment date near or slightly after the latest due date"""
        latest_due_date = max(fattura.data_scadenza for fattura in fatture)
        
        # Payment occurs 0-15 days after the latest due date
        days_after = random.randint(0, 15)
        return latest_due_date + timedelta(days=days_after)
    
    def _generate_group_payment_reference(self, fatture: List[Fattura], 
                                        include_all_ids: bool = True) -> Tuple[str, str]:
        """Generate payment details and causale with invoice references"""
        if include_all_ids and len(fatture) <= 3:
            # Include all invoice numbers for small groups
            invoice_numbers = [f.numero_fattura for f in fatture]
            invoice_ref = ", ".join(invoice_numbers)
            
            dettaglio = f"BONIFICO SEPA - Pagamento fatture n. {invoice_ref}"
            causale = f"Pagamento fatture {invoice_ref}"
        elif len(fatture) > 3:
            # For larger groups, use period reference
            first_date = min(f.data_emissione for f in fatture)
            last_date = max(f.data_emissione for f in fatture)
            
            if first_date.month == last_date.month:
                period_ref = first_date.strftime("%m/%Y")
                dettaglio = f"BONIFICO SEPA - Pagamento fatture periodo {period_ref}"
                causale = f"Pagamento fatture {period_ref}"
            else:
                period_ref = f"{first_date.strftime('%m/%Y')}-{last_date.strftime('%m/%Y')}"
                dettaglio = f"BONIFICO SEPA - Pagamento fatture periodo {period_ref}"
                causale = f"Pagamento fatture {period_ref}"
        else:
            # Generic group payment reference
            dettaglio = f"BONIFICO SEPA - Pagamento fatture multiple"
            causale = f"Pagamento fatture multiple"
        
        return dettaglio, causale
    
    def generate_scenario_n_1_group_payment(self, n_groups: int) -> Tuple[List[Fattura], List[Transazione], List[GroundTruth]]:
        """Generate N:1 group payment scenario"""
        fatture = []
        transazioni = []
        ground_truth = []
    
        # FIX: Use existing companies instead of generating new ones
        companies_to_use = self._select_companies_for_scenario(n_groups)
    
        for company in companies_to_use:
            # Generate a group of invoices for the same company
            n_invoices = np.random.choice([2, 3, 4, 5], p=[0.4, 0.3, 0.2, 0.1])
            
            # Choose billing period type
            billing_period = random.choice(["monthly", "quarterly", "weekly"])
            
            # Generate group of invoices within same billing period
            group_fatture = self._generate_billing_period_invoices(
                company, n_invoices, billing_period
            )
            fatture.extend(group_fatture)
            
            # Calculate total amount for group payment
            total_amount = sum(f.importo for f in group_fatture)
            
            # Add small variation to total (±2% for rounding, fees, etc.)
            payment_amount = total_amount * random.uniform(0.98, 1.02)
            
            # Generate payment date based on latest due date
            payment_date = self._generate_group_payment_date(group_fatture)
            
            # Generate payment reference including invoice information
            dettaglio, causale = self._generate_group_payment_reference(group_fatture)
            
            # Create group payment transaction
            transazione = Transazione(
                data=payment_date,
                dettaglio=dettaglio,
                importo=payment_amount,
                tipologia_movimento="pagamento",
                controparte=company['nome'],
                causale=causale,
                invoice_number=1  # Indicates reference to invoice numbers
            )
            transazioni.append(transazione)
            
            # Create ground truth entries for each invoice in the group
            for i, fattura in enumerate(group_fatture):
                # Calculate proportional amount covered
                amount_covered = (fattura.importo / total_amount) * payment_amount
                
                gt = GroundTruth(
                    fattura_id=str(fattura.id),
                    pagamento_id=str(transazione.id),
                    match_type=MatchType.PARTIAL.value,
                    confidence=0.85,  # Slightly lower confidence for group payments
                    amount_covered=amount_covered,
                    notes=f"Group payment {i+1}/{n_invoices} - {billing_period} billing"
                )
                ground_truth.append(gt)
        
        return fatture, transazioni, ground_truth
    
    def generate_scenario_1_n_installments(self, n_invoices: int) -> Tuple[List[Fattura], List[Transazione], List[GroundTruth]]:
        """Generate 1:N installment scenario"""
        fatture = []
        transazioni = []
        ground_truth = []
    
        companies_to_use = self._select_companies_for_scenario(n_invoices)

        for batch in self._chunk_list(companies_to_use, self.batch_size):
            batch_fatture = self._generate_invoices_batch(
                batch,
                amount_range=(2000, 15000),
                scenario_type="installment",
                use_recurrency=True,
            )
            fatture.extend(batch_fatture)

            for fattura, company in zip(batch_fatture, batch):
                n_installments = random.choice([2, 3, 4])
                installment_amount = round(fattura.importo / n_installments, 2)
                for j in range(n_installments):
                    transazione = self.generate_payment(
                        fattura,
                        company,
                        AmountPattern.EXACT,
                        TimingPattern.STANDARD,
                        QualityLevel.NOISY,
                    )
                    transazione.importo = installment_amount
                    transazione.data = fattura.data_emissione + timedelta(days=30 * (j + 1))
                    transazioni.append(transazione)
                    ground_truth.append(
                        GroundTruth(
                            fattura_id=str(fattura.id),
                            pagamento_id=str(transazione.id),
                            match_type=MatchType.EXACT.value,
                            confidence=1.0,
                            amount_covered=transazione.importo,
                            notes=f"Installment {j+1}/{n_installments} for invoice {fattura.id}",
                        )
                    )
        return fatture, transazioni, ground_truth

    def generate_scenario_standalone_invoices(self, n_invoices: int) -> Tuple[List[Fattura], List[Transazione], List[GroundTruth]]:
        """Generate a number of standalone invoices (without payments)."""
        fatture = []
        transazioni = []  # No transactions for standalone invoices
        ground_truth = [] # No ground truth for standalone invoices
    
        companies_to_use = self._select_companies_for_scenario(n_invoices)

        for batch in self._chunk_list(companies_to_use, self.batch_size):
            batch_fatture = self._generate_invoices_batch(
                batch,
                scenario_type="standalone_invoice",
                use_recurrency=False,
            )
            fatture.extend(batch_fatture)
    
        return fatture, transazioni, ground_truth

    def generate_scenario_standalone_payments(self, n_payments: int) -> Tuple[List[Fattura], List[Transazione], List[GroundTruth]]:
        """Generate a number of standalone payments (without invoices)."""
        fatture = []  # No invoices for standalone payments
        transazioni = []
        ground_truth = [] # No ground truth for standalone payments
    
        companies_to_use = self._select_companies_for_scenario(n_payments)

        for batch in self._chunk_list(companies_to_use, self.batch_size):
            dummy_invoices = []
            for company in batch:
                data_emissione = self.fake.date_between(start_date='-1y', end_date='today')
                data_scadenza = data_emissione + timedelta(days=random.randint(30, 90))
                dummy_invoices.append(
                    Fattura(
                        id=uuid.uuid4(),
                        data_emissione=data_emissione,
                        data_scadenza=data_scadenza,
                        numero_fattura="DUMMY",
                        descrizione="Dummy invoice for standalone payment generation",
                        importo=random.uniform(50, 5000),
                        prestatore=company['nome'],
                        committente=self.fake.company(),
                    )
                )

            batch_trans = self._generate_payments_batch(
                dummy_invoices,
                batch,
                [AmountPattern.EXACT] * len(batch),
                [TimingPattern.STANDARD] * len(batch),
                [QualityLevel.NOISY] * len(batch),
            )
            transazioni.extend(batch_trans)
    
        return fatture, transazioni, ground_truth

    def generate_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """Generate complete synthetic dataset"""
        all_fatture = []
        all_transazioni = []
        all_ground_truth = []
        
        # Generate different scenarios based on config
        scenarios_config = self.config.get('scenarios', {})
        
        # 1:1 Perfect Match
        if scenarios_config.get('perfect_1_1', 0) > 0:
            fatture, transazioni, gt = self.generate_scenario_1_1_perfect(
                scenarios_config['perfect_1_1']
            )
            all_fatture.extend(fatture)
            all_transazioni.extend(transazioni)
            all_ground_truth.extend(gt)
        
        # 1:N Installments
        if scenarios_config.get('installments_1_n', 0) > 0:
            fatture, transazioni, gt = self.generate_scenario_1_n_installments(
                scenarios_config['installments_1_n']
            )
            all_fatture.extend(fatture)
            all_transazioni.extend(transazioni)
            all_ground_truth.extend(gt)

        # N:1 Group Payment
        if scenarios_config.get('group_payment_n_1', 0) > 0:
            fatture, transazioni, gt = self.generate_scenario_n_1_group_payment(
                scenarios_config['group_payment_n_1']
            )
            all_fatture.extend(fatture)
            all_transazioni.extend(transazioni)
            all_ground_truth.extend(gt)

        # Standalone Invoices
        if scenarios_config.get('standalone_invoices', 0) > 0:
            fatture, transazioni, gt = self.generate_scenario_standalone_invoices(
                scenarios_config['standalone_invoices']
            )
            all_fatture.extend(fatture)
            all_transazioni.extend(transazioni)
            all_ground_truth.extend(gt)

        # Standalone Payments
        if scenarios_config.get('standalone_payments', 0) > 0:
            fatture, transazioni, gt = self.generate_scenario_standalone_payments(
                scenarios_config['standalone_payments']
            )
            all_fatture.extend(fatture)
            all_transazioni.extend(transazioni)
            all_ground_truth.extend(gt)
        
        # Convert to DataFrames
        fatture_df = pd.DataFrame([fattura.model_dump() for fattura in all_fatture])
        transazioni_df = pd.DataFrame([transazione.model_dump() for transazione in all_transazioni])
        ground_truth_df = pd.DataFrame([asdict(gt) for gt in all_ground_truth])
        
        # Generate metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'total_fatture': len(all_fatture),
            'total_transazioni': len(all_transazioni),
            'total_matches': len(all_ground_truth),
            'scenarios': scenarios_config,
            'config': self.config
        }
        
        return fatture_df, transazioni_df, ground_truth_df, metadata
    
    def export_dataset(self, dataset: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict], 
                      output_dir: str = "output"):
        """Export dataset to multiple formats"""
        fatture_df, transazioni_df, ground_truth_df, metadata = dataset
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export to CSV
        fatture_df.to_csv(f"{output_dir}/invoices.csv", index=False)
        transazioni_df.to_csv(f"{output_dir}/payments.csv", index=False)
        ground_truth_df.to_csv(f"{output_dir}/ground_truth.csv", index=False)
        
        # Export metadata
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Dataset exported to {output_dir}")