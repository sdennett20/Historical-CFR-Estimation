# Data Dictionary

Describes each dataset in `Data/` (on the `data_sourcing` branch): its source, what it records, and a per-column description.

---

## 1. `rosello2015_supplementary1.csv`

**Source:** Rosello et al., 2015 — supplementary data  
**Citation:** Rosello et al., ‘Ebola Virus Disease in the Democratic Republic of the Congo, 1976-2014’.
**Link/DOI:** 10.7554/eLife.09015  
**Outbreak(s):** Multiple historical EVD / Bundibugyo virus disease outbreaks  
**Unit of observation:** Individual case (one row per person)
**Description:** A line list for all outbreaks in the Democratic Republic of the Congo since 1976, comprising 996 cases.

| Column | Description |
|---|---|
| `Outbreak` | Name/label of the outbreak |
| `Year_outbreak` | Year the outbreak occurred |
| `Person_ID` | Unique case identifier |
| `Age` | Age of the case |
| `Sex` | Sex of the case |
| `Date_of_onset_symp` | Date of symptom onset |
| `Date_hospital_discharge` | Date discharged from hospital |
| `Outcome` | Case outcome (e.g. died / survived) |
| `Case_definition` | Classification of the case (confirmed / probable / suspected) |
| `Date_of_notification` | Date case was notified to authorities |
| `Date_of_Hospitalisation` | Date of hospitalisation |
| `Date_disease_ended` | Date the disease episode ended |
| `Date_of_Death` | Date of death (if applicable) |
| `Occupation` | Occupation of the case |
| `Fever` | Presence of fever (1/0/NA) |
| `Diarrhea` | Presence of diarrhoea (1/0/NA) |
| `Abdominal_pain` | Presence of abdominal pain (1/0/NA) |
| `Headache` | Presence of headache (1/0/NA) |
| `Vomiting` | Presence of vomiting (1/0/NA) |
| `Haemorrhagic_symptoms` | Presence of haemorrhagic symptoms (1/0/NA) |
| `Hiccup` | Presence of hiccups (1/0/NA) |
| `Info_one_clincal_symptom` | Whether at least one clinical symptom was recorded (1/0) |

---

## 2. `ebola_sierraleone_2014_confirmed.csv`

**Source:** 
**Citation:** <!-- full citation here -->  
**Link/DOI:** <!-- link here -->  
**Outbreak:** Sierra Leone, 2014 West Africa outbreak  
**Unit of observation:** Individual confirmed case

| Column | Description |
|---|---|
| `ID` | Unique case identifier |
| `Name` | Name of the case |
| `Age` | Age of the case |
| `Sex` | Sex of the case |
| `Date of symptom onset` | Date of symptom onset |
| `Date of sample tested` | Date the diagnostic sample was tested |
| `District` | Administrative district of the case |
| `Chiefdom` | Chiefdom (sub-district administrative unit) |

---

## 3. `ebola_sierraleone_2014_suspected.csv`

**Source:** <!-- source here -->  
**Citation:** <!-- full citation here -->  
**Link/DOI:** <!-- link here -->  
**Outbreak:** Sierra Leone, 2014 West Africa outbreak  
**Unit of observation:** Individual suspected case  
**Note:** Same column structure as confirmed cases file above.

| Column | Description |
|---|---|
| `ID` | Unique case identifier |
| `Name` | Name of the case |
| `Age` | Age of the case |
| `Sex` | Sex of the case |
| `Date of symptom onset` | Date of symptom onset |
| `Date of sample tested` | Date the diagnostic sample was tested |
| `District` | Administrative district of the case |
| `Chiefdom` | Chiefdom (sub-district administrative unit) |

---

## 4. `Uganda2022globaldothealth.csv`

**Source:** Global.health line-list database from Github Repository
**Citation:** Global.health Ebola (accessed on 2026-06-30)
**Link/DOI:** https://github.com/globaldothealth/ebola/tree/main
**Outbreak:** Uganda, 2022  
**Unit of observation:** Individual case

| Column | Description |
|---|---|
| `ID` | Unique case identifier |
| `Pathogen` | Pathogen name |
| `Case_status` | Case classification (confirmed / probable / suspected) |
| `Location_District` | District of the case |
| `Country` | Country |
| `Age` | Age of the case |
| `Gender` | Gender of the case |
| `Occupation` | Occupation |
| `Healthcare_worker` | Whether the case is a healthcare worker (Y/N) |
| `Symptoms` | Symptoms reported |
| `Date_onset` | Date of symptom onset |
| `Date_confirmation` | Date of case confirmation |
| `Confirmation_method` | Method used to confirm the case |
| `Previous_infection` | History of prior infection (Y/N) |
| `Co_infection` | Co-infection present (Y/N) |
| `Pre_existing_condition` | Pre-existing medical conditions |
| `Pregnancy_status` | Pregnancy status |
| `Vaccination` | Whether vaccinated (Y/N) |
| `Vaccine_name` | Name of vaccine received |
| `Vaccine_date` | Date of vaccination |
| `Vaccine_side_effects` | Reported vaccine side effects |
| `Date_of_first_consult` | Date of first clinical consultation |
| `Hospitalised` | Whether hospitalised (Y/N) |
| `Reason_for_hospitalization` | Reason for hospitalisation |
| `Date_hospitalisation` | Date of hospitalisation |
| `Date_discharge_hospital` | Date of hospital discharge |
| `Intensive_care (Y/N/NA)` | Whether admitted to intensive care |
| `Date_admission_ICU` | Date of ICU admission |
| `Date_discharge_ICU` | Date of ICU discharge |
| `Home_monitoring (Y/N/NA)` | Whether under home monitoring |
| `Isolated (Y/N/NA)` | Whether isolated |
| `Date_isolation` | Date isolation began |
| `Outcome` | Case outcome (died / recovered) |
| `Date_Death` | Date of death (if applicable) |
| `Date_Recovered` | Date of recovery (if applicable) |
| `Contact_with_case` | Whether contact with a known case (Y/N) |
| `Contact_ID` | ID of the contact case |
| `Contact_setting` | Setting of contact exposure |
| `Contact_setting_other` | Other contact setting details |
| `Contact_animal` | Whether animal contact reported |
| `Contact_comment` | Free-text contact notes |
| `Transmission` | Transmission route |
| `Travel_history (Y/N/NA)` | Whether travel history present |
| `Travel_history_entry` | Entry point of travel |
| `Travel_history_start` | Start date of travel |
| `Travel_history_location` | Travel location |
| `Travel_history_country` | Travel country |
| `Genomica_Metadata` | Genomic metadata reference |
| `Accession Number` | Sequence accession number |
| `Source` | Primary data source |
| `Source_II` | Secondary data source |
| `Date_entry` | Date entered into database |
| `Date_last_modified` | Date record last modified |
| `Source_III` | Tertiary data source |
| `Source_IV` | Quaternary data source |
| `Source_V` | Fifth data source |
| `Source_VI` | Sixth data source |

---

## 5. `DRC2018_humdata_MOH-Total.csv`

**Source:** DRC Ministry of Health, via Humanitarian Data Exchange (HDX / humdata.org)  
**Citation:** DRC Ministry of Health, World Health Organization (WHO). Ebola Cases and Deaths — DRC North Kivu. Humanitarian Data Exchange (HDX). Available at: https://data.humdata.org/dataset/ebola-cases-and-deaths-drc-north-kivu. Accessed: 1 July 2026. 
**Link/DOI:** [<!-- link here --> ](https://data.humdata.org/m/dataset/ebola-cases-and-deaths-drc-north-kivu) 
**Outbreak:** DRC, 2018  
**Unit of observation:** National daily/periodic situation report total

| Column | Description |
|---|---|
| `publication_date` | Date the situation report was published |
| `report_date` | Date the report data refers to |
| `country` | Country (DRC) |
| `confirmed_cases` | Cumulative confirmed cases |
| `probable_cases` | Cumulative probable cases |
| `total_cases` | Cumulative total cases (confirmed + probable) |
| `confirmed_deaths` | Cumulative confirmed deaths |
| `total_deaths` | Cumulative total deaths |
| `total_suspected_cases` | Cumulative suspected cases |
| `new_deaths` | New deaths since last report |
| `new_cured` | New recoveries since last report |
| `total_cured` | Cumulative recoveries |
| `new_suspected_cases` | New suspected cases since last report |
| `old_suspected_cases` | Suspected cases from prior periods |
| `confirmed_cases_change` | Change in confirmed cases since last report |
| `probable_cases_change` | Change in probable cases since last report |
| `total_cases_change` | Change in total cases since last report |
| `confirmed_deaths_change` | Change in confirmed deaths since last report |
| `total_deaths_change` | Change in total deaths since last report |
| `total_suspected_cases_change` | Change in suspected cases since last report |
| `source` | Source URL or reference for this report |

---

## 6. `DRC2018_humdata_MOH-By-Health-Zone.csv`

**Source:** DRC Ministry of Health, via Humanitarian Data Exchange (HDX / humdata.org)  
**Citation:** DRC Ministry of Health, World Health Organization (WHO). Ebola Cases and Deaths — DRC North Kivu. Humanitarian Data Exchange (HDX). Available at: https://data.humdata.org/dataset/ebola-cases-and-deaths-drc-north-kivu. Accessed: 1 July 2026. 
**Link/DOI:** https://data.humdata.org/m/dataset/ebola-cases-and-deaths-drc-north-kivu 
**Outbreak:** DRC, 2018  
**Unit of observation:** Health-zone-level daily/periodic situation report  
**Note:** Same fields as the national totals file above, with the addition of geographic columns.

| Column | Description |
|---|---|
| `publication_date` | Date the situation report was published |
| `report_date` | Date the report data refers to |
| `country` | Country (DRC) |
| `province` | Province within DRC |
| `health_zone` | Health zone (sub-provincial administrative unit) |
| `confirmed_cases` | Cumulative confirmed cases in this health zone |
| `probable_cases` | Cumulative probable cases |
| `total_cases` | Cumulative total cases |
| `confirmed_deaths` | Cumulative confirmed deaths |
| `total_deaths` | Cumulative total deaths |
| `total_suspected_cases` | Cumulative suspected cases |
| `new_deaths` | New deaths since last report |
| `new_cured` | New recoveries since last report |
| `total_cured` | Cumulative recoveries |
| `new_suspected_cases` | New suspected cases since last report |
| `old_suspected_cases` | Suspected cases from prior periods |
| `confirmed_cases_change` | Change in confirmed cases since last report |
| `probable_cases_change` | Change in probable cases since last report |
| `total_cases_change` | Change in total cases since last report |
| `confirmed_deaths_change` | Change in confirmed deaths since last report |
| `total_deaths_change` | Change in total deaths since last report |
| `total_suspected_cases_change` | Change in suspected cases since last report |
| `source` | Source URL or reference for this report |

---

## 7. `drc_ebola_cases_consolidated.csv`

**Source:** <!-- source here -->  
**Citation:** <!-- full citation here -->  
**Link/DOI:** <!-- link here -->  
**Outbreak:** DRC (consolidated across outbreaks)  
**Unit of observation:** Aggregate measure per location/time period

| Column | Description |
|---|---|
| `location_country` | Country |
| `location_level` | Administrative level of the location |
| `location_name` | Name of the location |
| `location_name_source` | Source of the location name |
| `location_code` | Location code |
| `location_code_type` | Type of location code (e.g. PCODE, ISO) |
| `reference_date` | Date the data refers to |
| `measure` | Type of measure recorded (e.g. cases, deaths) |
| `case_classification` | Classification (confirmed / probable / suspected) |
| `time_period` | Time period the value covers |
| `value` | Numeric value of the measure |
| `unit` | Unit of the value |
| `source` | Data source name |
| `source_url` | URL of the data source |
| `source_indicator_name` | Name of the indicator as given by the source |
| `indicator_label` | Standardised indicator label |
| `reporting_period_start` | Start of the reporting period |
| `reporting_period_end` | End of the reporting period |
| `last_updated` | Date the record was last updated |
| `data_quality_flag` | Any data quality flags or warnings |
| `notes` | Free-text notes |
