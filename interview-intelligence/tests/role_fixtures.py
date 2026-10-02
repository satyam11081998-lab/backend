"""Eight role families x (CV, JD, a family-appropriate answer). Synthetic people, .invalid emails."""

ROLES = {
    "marketing": dict(
        cv="""Asha Rao
asha@example.invalid
Assistant Brand Manager, Acme Foods Jun 2022 - Present
- Led the repositioning of a biscuit brand for evening snacking, growing share 1.4 points in a year
- Built a media-mix model that moved 20% of spend from TV to digital video
Marketing Associate, Zeta Retail Jan 2020 - May 2022
- Analysed panel data to find a lapsed-buyer segment worth 8% of volume
Skills: brand management, consumer insight, media planning, Excel""",
        jd="""Brand Manager - Biscuits
Company: Nimbus Consumer Products
Own the brand P&L and annual plan
Lead consumer insight work and translate it into positioning
Requirements
3+ years of brand management experience in FMCG (required)
Strong understanding of positioning, pricing and media
Experience with campaign analytics and agencies""",
        answer=("I led the repositioning myself because household panel data showed 60% of consumption happened after "
                "6pm. I decided to shift the message to evening snacking and moved 20% of media to digital video, "
                "which meant giving up some TV reach. Share grew 1.4 points in a year against a control market."),
    ),
    "finance_corporate": dict(
        cv="""Rohan Mehta
rohan@example.invalid
Financial Analyst, Orion Industries Jul 2021 - Present
- Built the driver-based FP&A model used for the annual budget across 4 business units
- Cut month-end close from 9 to 6 days by automating reconciliations
Audit Associate, Big Firm LLP Aug 2019 - Jun 2021
- Audited revenue recognition for 3 listed clients
Skills: FP&A, financial modelling, variance analysis, Excel, SQL""",
        jd="""Senior Financial Analyst - FP&A
Company: Helix Manufacturing
Own the annual budget and monthly forecast
Partner with business heads on variance analysis
Requirements
4+ years of FP&A or corporate finance experience (required)
Strong financial modelling and Excel skills
Knowledge of working capital and cash flow forecasting""",
        answer=("I built the forecast model myself because the old one missed working capital swings. I decided to "
                "drive revenue from order intake and receivable days, which meant we caught a 12 crore cash gap two "
                "months early. Forecast error fell from 9% to 3% across four quarters, measured against actuals."),
    ),
    "consulting": dict(
        cv="""Neha Kapoor
neha@example.invalid
Associate Consultant, Apex Strategy Partners Jul 2021 - Present
- Led a cost diagnostic for a retail client that identified 14% savings in logistics
- Built the market-sizing model for a fintech entry case
Business Analyst, DataWorks Jun 2019 - Jun 2021
- Analysed churn drivers for a telecom client
Skills: problem structuring, market sizing, Excel, PowerPoint""",
        jd="""Consultant - Strategy Practice
Company: Meridian Advisory
Structure ambiguous client problems and drive analysis
Lead workstreams and client conversations
Requirements
2+ years of consulting experience (required)
Strong problem structuring and hypothesis-driven analysis
Ability to communicate synthesis to senior clients""",
        answer=("I structured the diagnostic myself into network, inventory and transport because those covered 90% "
                "of logistics cost. I decided to test the transport hypothesis first, which meant pausing the "
                "inventory analysis. We found 14% savings and the client validated 11% in the first quarter."),
    ),
    "software_engineering": dict(
        cv="""Karthik Iyer
karthik@example.invalid
Software Engineer, CloudCart Jan 2021 - Present
- Built the order service in Go handling 3,000 requests per second
- Reduced p99 latency from 900ms to 250ms by redesigning the caching layer
Software Engineer Intern, ByteLabs May 2020 - Dec 2020
- Developed an internal dashboard in React
Skills: Go, Python, PostgreSQL, Redis, Kubernetes, system design""",
        jd="""Senior Software Engineer - Backend
Company: Quanta Payments
Design and build high-throughput backend services
Own reliability and performance of payment APIs
Requirements
4+ years of backend development experience (required)
Strong knowledge of distributed systems and databases
Experience with Go or Java and cloud infrastructure""",
        answer=("I redesigned the caching layer myself because 70% of reads hit the same product keys. I chose a "
                "read-through cache with request coalescing instead of a write-through design, which meant accepting "
                "a few seconds of staleness. p99 latency fell from 900ms to 250ms, measured over two weeks of traffic."),
    ),
    "sales": dict(
        cv="""Vikram Singh
vikram@example.invalid
Area Sales Manager, FreshFoods Apr 2021 - Present
- Led a team of 8 sales officers across 1,200 outlets, growing secondary sales 22%
- Opened 300 new outlets in two quarters through a distributor incentive scheme
Sales Officer, DairyBest Jun 2018 - Mar 2021
- Grew territory volume 15% year on year
Skills: distribution, trade marketing, negotiation, Excel""",
        jd="""Regional Sales Manager
Company: Sunrise Beverages
Own regional sales targets and distributor network
Lead and coach area sales managers
Requirements
5+ years of FMCG sales experience (required)
Strong distributor management and negotiation skills
Experience leading field sales teams""",
        answer=("I led the expansion myself because 40% of our beat outlets were not stocking the new range. I decided "
                "to pay distributors a per-outlet activation incentive instead of a flat discount, which meant a higher "
                "cost per outlet. We opened 300 outlets in two quarters and secondary sales grew 22%."),
    ),
    "product_management": dict(
        cv="""Meera Nair
meera@example.invalid
Product Manager, ShopEase Mar 2021 - Present
- Owned checkout; launched one-tap UPI payments that lifted conversion 6%
- Ran 14 A/B tests in a year with a shared experiment framework
Associate Product Manager, TravelNow Jul 2019 - Feb 2021
- Shipped a fare-alert feature used by 200,000 users
Skills: product discovery, A/B testing, SQL, roadmapping""",
        jd="""Senior Product Manager - Payments
Company: Lumen Fintech
Own the payments roadmap and outcomes
Work with engineering, design and risk teams
Requirements
4+ years of product management experience (required)
Strong understanding of experimentation and metrics
Ability to prioritise with incomplete data""",
        answer=("I owned the checkout redesign because drop-off at payment was 38%. I decided to ship one-tap UPI "
                "first instead of saved cards, which meant delaying a partner request. I validated it with an A/B "
                "test on 10% of traffic and conversion rose 6% with no rise in payment failures."),
    ),
    "operations": dict(
        cv="""Suresh Patel
suresh@example.invalid
Operations Manager, QuickShip Logistics Feb 2020 - Present
- Ran a 24x7 fulfilment centre with 120 staff, improving on-time dispatch from 88% to 97%
- Reduced cost per order 15% through shift redesign and slotting changes
Shift Supervisor, MegaMart Jun 2017 - Jan 2020
- Cut picking errors by 40% with a barcode check process
Skills: lean, process improvement, workforce planning, Excel""",
        jd="""Senior Operations Manager - Fulfilment
Company: Atlas Commerce
Own fulfilment centre performance, cost and safety
Lead process improvement and workforce planning
Requirements
6+ years of operations experience (required)
Strong process improvement and lean knowledge
Experience managing large frontline teams""",
        answer=("I redesigned the shifts myself because 60% of orders arrived after 6pm while staffing was flat. I "
                "decided to move 30 people to a late shift and re-slot fast movers near packing, which meant "
                "renegotiating rosters. On-time dispatch rose from 88% to 97% and cost per order fell 15%."),
    ),
    "human_resources": dict(
        cv="""Anita Desai
anita@example.invalid
HR Business Partner, Novus Tech Aug 2020 - Present
- Partnered with 3 engineering leaders across 400 employees; cut regretted attrition from 18% to 11%
- Designed a manager capability programme completed by 60 managers
Talent Acquisition Specialist, HireRight Jun 2018 - Jul 2020
- Closed 120 technical hires a year
Skills: HR business partnering, talent management, employee relations""",
        jd="""HR Business Partner - Technology
Company: Vertex Software
Partner with technology leaders on people strategy
Lead performance, talent and retention programmes
Requirements
5+ years of HR business partner experience (required)
Strong knowledge of talent management and employee relations
Ability to influence senior leaders""",
        answer=("I led the retention work myself because exit data showed 70% of regretted leavers cited their manager. "
                "I decided to fund a manager capability programme instead of a blanket retention bonus, which meant "
                "a slower payoff. Regretted attrition fell from 18% to 11% within a year."),
    ),
}

# family -> (mode, difficulty, minutes): every family is run in a different configuration.
PLANS = {
    "marketing": ("cv_jd", "medium", 45),
    "finance_corporate": ("technical_deep_dive", "hard", 45),
    "consulting": ("case", "expert", 30),
    "software_engineering": ("technical", "hard", 60),
    "sales": ("stress", "medium", 30),
    "product_management": ("hiring_manager", "medium", 45),
    "operations": ("grill", "grill", 30),
    "human_resources": ("hr_behavioral", "easy", 15),
}
