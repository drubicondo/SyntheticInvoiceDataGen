# Piano per Dataset Sintetico Riconciliazione Fatture-Pagamenti

## 1. Analisi delle Casistiche (Taxonomy)

### 1.1 Relazioni Cardinali
- **1:1** - Una fattura, un pagamento
- **1:N** - Una fattura, più pagamenti (pagamenti rateali/parziali)
- **N:1** - Più fatture, un pagamento (pagamento cumulativo)
- **N:M** - Più fatture, più pagamenti (scenario complesso)

### 1.2 Timing Patterns
- **Standard** - Pagamento 0-90 giorni dopo fattura
- **Ritardato** - Pagamento oltre 90 giorni
- **Anticipato** - Pagamento prima dell'emissione fattura
- **Contemporaneo** - Pagamento stesso giorno fattura

### 1.3 Amount Matching
- **Esatto** - Importo pagamento = importo fattura
- **Parziale** - Pagamento < fattura (con o senza completamento)
- **Eccedente** - Pagamento > fattura (credito/errore)
- **Con sconto** - Pagamento con sconto concordato
- **Con penali** - Pagamento con maggiorazioni per ritardo

### 1.4 Quality Patterns
- **Perfect Match** - Tutti i dati corrispondono perfettamente
- **Fuzzy Match** - Piccole variazioni nei nomi/riferimenti
- **Ambiguous** - Più possibili corrispondenze
- **Noisy** - Dati con errori tipografici/formattazione

## 2. Struttura Dati

### 2.1 Schema Fattura
```python
{
    "fattura_id": str,
    "numero_fattura": str,
    "data_emissione": datetime,
    "data_scadenza": datetime,
    "cliente_id": str,
    "cliente_nome": str,
    "cliente_piva": str,
    "importo_lordo": float,
    "importo_netto": float,
    "iva": float,
    "descrizione": str,
    "note": str,
    "stato": str  # emessa, pagata, parzialmente_pagata, scaduta
}
```

### 2.2 Schema Pagamento
```python
{
    "pagamento_id": str,
    "data_pagamento": datetime,
    "data_valuta": datetime,
    "importo": float,
    "mittente": str,
    "mittente_iban": str,
    "causale": str,
    "riferimento_fattura": str,  # può essere None o impreciso
    "metodo_pagamento": str,  # bonifico, carta, assegno, etc.
    "note_banca": str
}
```

### 2.3 Ground Truth Labels
```python
{
    "fattura_id": str,
    "pagamento_id": str,
    "match_type": str,  # exact, partial, related, unrelated
    "confidence": float,  # 0.0 - 1.0
    "amount_covered": float,
    "notes": str
}
```

## 3. Generazione Dataset per Blocchi

### 3.1 Blocco 1: Scenari Standard (40% del dataset)
- **1:1 Perfect Match** (15%)
  - Stesso importo, cliente, tempistiche normali
  - Causale chiara con numero fattura
  
- **1:1 con Variazioni** (15%)
  - Piccole differenze importi (arrotondamenti)
  - Variazioni nei nomi clienti
  - Causali con errori tipografici
  
- **Timing Variations** (10%)
  - Pagamenti anticipati
  - Pagamenti in ritardo
  - Pagamenti contemporanei

### 3.2 Blocco 2: Scenari Complessi (35% del dataset)
- **1:N - Pagamenti Rateali** (10%)
  - Fattura divisa in 2-5 pagamenti
  - Tempistiche regolari o irregolari
  
- **N:1 - Pagamenti Cumulativi** (15%)
  - 2-10 fatture pagate insieme
  - Causale che elenca più riferimenti
  
- **N:M - Scenari Misti** (10%)
  - Combinazioni complesse
  - Pagamenti parziali incrociati

### 3.3 Blocco 3: Scenari Ambigui (15% del dataset)
- **Matching Ambiguo**
  - Più fatture con importi simili
  - Causali generiche
  
- **Dati Incompleti**
  - Causali mancanti o incomplete
  - Informazioni cliente parziali

### 3.4 Blocco 4: Scenari di Disturbo (10% del dataset)
- **False Positives**
  - Pagamenti non correlati con causali simili
  - Importi casuali che corrispondono
  
- **Outliers**
  - Importi molto diversi
  - Tempistiche estreme

## 4. Strategia di Generazione

### 4.1 Generazione Strutturale (Python)
```python
class DatasetGenerator:
    def __init__(self, config):
        self.config = config
        self.ai_generator = AITextGenerator()
    
    def generate_block(self, block_type, size):
        # Genera struttura dati per ogni blocco
        pass
    
    def generate_companies(self, n):
        # Genera anagrafica clienti realistica
        pass
    
    def generate_temporal_patterns(self):
        # Genera pattern temporali realistici
        pass
```

### 4.2 Generazione Testuale (GenAI)
**Prompt Template per Causali:**
```
Genera una causale di pagamento realistica per:
- Cliente: {cliente_nome}
- Fattura: {numero_fattura} 
- Importo: {importo}
- Scenario: {scenario_type}
- Qualità: {quality_level} (perfect/fuzzy/noisy)

La causale deve essere tipica del settore italiano e includere variazioni naturali.
```

**Prompt Template per Descrizioni Fatture:**
```
Genera una descrizione fattura realistica per:
- Settore: {settore}
- Cliente: {cliente}
- Importo: {importo}
- Tipo servizio: {tipo_servizio}

Include dettagli specifici del settore e terminologia appropriata.
```

### 4.3 Pipeline di Generazione
1. **Setup Phase**
   - Definizione parametri per ogni blocco
   - Generazione anagrafica clienti base
   
2. **Structural Generation**
   - Creazione scheletro fatture/pagamenti
   - Definizione relazioni ground truth
   
3. **AI Text Generation**
   - Generazione batch causali
   - Generazione descrizioni
   - Applicazione noise patterns
   
4. **Validation & Quality Control**
   - Verifica consistenza dati
   - Bilanciamento classi
   - Export in formati multipli

## 5. Implementazione Tecnica

### 5.1 Tecnologie Suggerite
- **Core**: Python, Pandas, Faker
- **GenAI**: OpenAI API, Anthropic Claude, o modelli locali
- **Graph**: NetworkX per strutture grafo
- **Export**: JSON, CSV, Parquet
- **Visualization**: Matplotlib, Plotly per analisi

### 5.2 Configurazione Modulare
```python
CONFIG = {
    "blocks": {
        "standard": {"size": 4000, "variations": 3},
        "complex": {"size": 3500, "max_relations": 10},
        "ambiguous": {"size": 1500, "noise_level": 0.3},
        "outliers": {"size": 1000, "extremes": True}
    },
    "ai_generation": {
        "model": "claude-sonnet-4",
        "batch_size": 100,
        "quality_levels": ["perfect", "fuzzy", "noisy"]
    }
}
```

### 5.3 Metriche di Qualità
- **Coverage**: Percentuale casistiche coperte
- **Balance**: Distribuzione equilibrata scenari
- **Realism**: Validazione manuale campioni
- **Complexity**: Gradiente difficoltà appropriato

## 6. Output e Validazione

### 6.1 Formati Export
- **Training Set**: 70% con ground truth completa
- **Validation Set**: 15% per tuning iperparametri
- **Test Set**: 15% per valutazione finale
- **Holdout Set**: Scenari edge case per stress testing

### 6.2 Documentazione Dataset
- **Metadata**: Statistiche per ogni blocco
- **Schema**: Documentazione completa campi
- **Examples**: Campioni rappresentativi
- **Benchmarks**: Metriche baseline attese

## 7. Estensibilità

### 7.1 Parametri Configurabili
- Distribuzione scenari per settore
- Pattern temporali personalizzabili
- Livelli di noise controllabili
- Integrazione nuove casistiche

### 7.2 Feedback Loop
- Analisi performance modello per scenario
- Identificazione gap nel dataset
- Generazione iterativa miglioramenti