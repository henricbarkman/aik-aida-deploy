# Aidas metod

Den här metodbeskrivningen förklarar hur Aida räknar, vilka källor som används och vad som inte räknas. Aida hänvisar till den när någon frågar hur verktyget fungerar.

## Vad Aida räknar ut

Du beskriver projektet med egna ord, till exempel "byta golv i korridoren, 240 m²". Aida delar upp beskrivningen i byggdelar med mängd och enhet och räknar sedan fram följande för varje byggdel:

- **Baslinjen**: klimatpåverkan om byggdelen utförs med konventionella material utan särskild klimathänsyn.
- **Alternativen**: verkliga produkter med miljödeklaration (EPD) och deras klimatpåverkan för samma mängd.
- **Skillnaden**: hur mycket ett alternativ minskar eller ökar klimatpåverkan jämfört med baslinjen, och vad det kostar.

Resultatet är ett underlag för beslut, inte ett klimatbokslut.

## Klimatmåttet

Aida räknar GWP-fossil för livscykelskedena A1-A3, alltså produktskedet: råvaror, transport till fabriken och tillverkning. Det är samma mått som Boverkets klimatdatabas använder. Upptag av biogent kol räknas inte av, varken i baslinjen eller i alternativen.

Några miljödeklarationer är motsägelsefulla: det redovisade fossilvärdet stämmer inte med deklarationens egen totalsumma, eller värdet för markanvändning är större än fossilvärdet, vilket betyder att siffrorna hamnat på fel rad i den digitala deklarationen. För dem används GWP-GHG i stället, alltså totala växthusgasutsläpp exklusive biogent kol, och rapporten redovisar vilka poster det gäller. GWP-GHG ligger nära GWP-fossil i storlek men är inte samma indikator.

För de flesta kategorier har Aida ett rimligt intervall för klimatpåverkan per enhet, och klimatvärden prövas mot det. Ett värde utanför intervallet märks för kontroll. Ett värde som ligger mer än tre gånger över intervallets högsta värde, eller under en tredjedel av det lägsta, ersätts med intervallets mitt.

## Baslinjen

Baslinjen följer principen i NollCO2, Sweden Green Building Councils metod: den beskriver ett standardfall, alltså vad byggdelen brukar innebära klimatmässigt när den utförs på det sätt som är typiskt i dag. Den är inte ett värsta fall. Aida förenklar metoden genom att sätta ett värde per byggdel i stället för att modellera en hel typbyggnad.

Baslinjen sätts utifrån byggnadstyp och funktion: vad som är konventionellt för en sådan byggdel, inte vad projektet tänker välja. Följde baslinjen projektets eget materialval skulle jämförelsen bli cirkulär.

### Boverkets klimatdatabas först

Aida matchar varje byggdel mot Boverkets klimatdatabas och använder databasens typiska värden (Typical) för A1-A3. En produkt i Boverket används bara när den är samma material som byggdelens standardmaterial, som gipsskiva mot gipsskiva eller betong mot betong. En produkt av annan typ lånas aldrig bara för att den delar basmaterial, till exempel takduk av PVC för ett vinylgolv. Boverket saknar bland annat golvbeläggning, sanitetsporslin, vitvaror och belysning som egna produkter.

### EPD-typvärde när Boverket saknar materialet

Saknar Boverket materialet används ett typvärde ur Aidas EPD-katalog. Typvärdet är medianen av den övre halvan av katalogens värden i kategorin, ungefär den 75:e percentilen. Det motsvarar ett konventionellt val, inte det bästa på marknaden. Ett typvärde kräver minst fem produkter med miljödeklaration, för toaletter fyra.

En produkt räknas en gång även när tillverkaren har deklarerat den för flera fabriker. Mapelastic Zero finns till exempel som nio deklarationer, en per fabrik, och vägde tidigare som nio produkter i medianen. Nu räknas de som en produkt med medianen av fabrikernas värden. Samma namn från samma fabrik i flera värden, som olika storlekar av ett badrumsskåp, räknas som olika produkter. Det ändrade fem typvärden (1 oktober 2026): tätskikt per kg 1,06 till 1,60, hiss per kg 5,43 till 6,42, regelstål per kg 3,20 till 3,02, konstruktionsstål per kg 2,62 till 2,63, och innervägg per m² 3,80 till 4,10 (där för att de europeiska deklarationerna nu räknas per koncern, se nästa stycke).

Ett typvärde publiceras inte när en och samma koncern står bakom 60 procent eller mer av produkterna. Då vore typvärdet den koncernens sortiment och inte ett typiskt val. Koncernen räknas med dotterbolag, enligt bolagens egna uppgifter och årsredovisningar: Weber, Gyproc, Isover, Ecophon och Dahl hör till Saint-Gobain, Rockfon till Rockwool, JKE Design och Ballingslöv till Ballingslöv International. I stället gör Aida en uppskattning och skriver i raden varför. Det gäller i dag bland annat bafflar (Ecophon), undertakens bärverk (Rockfon), köksluckor (Ballingslöv International), förvaring (AJ Produkter), golv per styck (Kingspans installationsgolv) och fasadskikt per m² (Saint-Gobains putser). Avjämningsmassa och lägenhetsaggregat hölls tillbaka på samma sätt fram till 1 oktober 2026, då deklarationer från fler tillverkare kom in (Kiilto, Sika, Mapei, Marlon, FB, Sto, Bostik och Sakret för avjämning; Systemair, Vallox och S&P för lägenhetsaggregat). För bafflar och bärverk finns i dag inga deklarationer per m² eller löpmeter från andra tillverkare att hämta.

Varje rad med typvärde säger vilka de två största leverantörerna är och hur många av produkterna de står för. När det som är kvar efter den största i sin tur till 60 procent kommer från en enda annan, vilar typvärdet i praktiken på två leverantörers sortiment, och då står det också i raden. Det gäller till exempel akustikplattor, där Ecophon och Rockfon står för nästan alla deklarationer. Sådana typvärden publiceras ändå. Två oberoende tillverkare är två oberoende underlag, och för akustikplattor är de två i stort sett hela marknaden, medan till exempel glaspartiernas två brittiska tillverkare inte är det. Den skillnaden går inte att läsa ut ur katalogen, så Aida säger hur det ligger till i stället för att gissa.

För golv väljs först en undertyp, till exempel vinyl eller linoleum, utifrån det standardmaterial Aida har antagit. Har undertypen för få deklarationer används hela golvkategorin.

Ventilationsaggregat räknas inte med ett typvärde per styck. Katalogens aggregat spänner från omkring 100 kg CO2e för ett litet lägenhetsaggregat till 24 500 kg för ett stort skolaggregat, och skillnaden är storleken. Aida räknar därför i första hand per luftflöde: 24 av katalogens 72 aggregat-EPD:er anger aggregatets nominella luftflöde, och deras värden per m³/h ger typvärdet, som multipliceras med aggregatets eget flöde. Flödet läses ur komponentens namn eller beskrivning och gissas aldrig. De övriga fyrtioåtta anger inget flöde för den deklarerade enheten och ingår inte (Flexits Nordic-serie hänvisar till tillverkarens webbplats, Acetecs EPD är ett snitt över en hel serie, Swegons äldre EPD för storlek 011/012, S&P:s SABIK, Systemairs SAVE-serie och Vallox anger inget flöde, och S&P:s NASHIRA bara ett högsta). För Flexits ProNordic-aggregat, som anger två kapaciteter, används den vid SFP 1,5 kW/(m³/s), den nivå BBR säger att man ska eftersträva när ett aggregat med värmeåtervinning byts ut (avsnitt 9:959, tabell 9:95). Swegons GOLD/SILVER C RX anger också två flöden, ett högsta luftflöde och ett dimensionerande luftflöde i EPD:ns driftscenario (vid SFP 1,6). Det dimensionerande används, eftersom det är det flöde deklarationen själv räknar aggregatet för; det högsta är kapaciteten vid högre SFP, samma val som för ProNordic. S&P:s PURECLASS, ett aggregat för ett enskilt klassrum, anger 700 m³/h som sin representativa driftpunkt. Swegons EPD:er deklarerar ett färdigt aggregat, men datafilen anger referensen som aggregatets vikt. Aida läser dem per aggregat, och bara när vikten i datafilen är densamma som i EPD:ns egen text. Saknas flödet men framgår det att aggregatet betjänar en hel byggnad (till exempel "centralt aggregat" eller en systembeteckning som LB01) används ett typvärde för byggnadsaggregat: ett aggregat med byggnadsaggregatens medianflöde, räknat med samma värde per luftflöde. I medianflödet räknas varje tillverkares serie en gång: först medianen av varje tillverkares flöden, sedan medianen av dem. Annars bestämmer den tillverkare som deklarerat flest storlekar vad ett typiskt aggregat är. När Swegon lade till nio storlekar flyttade den raka medianen från 3 700 till 6 000 m³/h utan att någon byggnad blivit större. Räknat per tillverkare blir den 5 910 m³/h och typvärdet 7 269 kg CO2e i stället för 7 380 (1 oktober 2026). Vilken klass en EPD tillhör har tagits ur EPD:ns egen beskrivning av användningsområdet, inte ur en påhittad gräns. Lägenhetsaggregat har ett eget typvärde per styck sedan 1 oktober 2026, från 39 deklarationer av fyra tillverkare (Systemair 20, Vallox 9, Flexit 8, S&P 2). Ingen av dem anger något flöde, så typvärdet är medianen av den övre halvan per aggregat. Vallox MyVallox-serie anges för både hem och andra byggnader och räknas därför i ingen av klasserna. Går varken flöde eller klass att avgöra gör Aida en uppskattning och skriver varför, och att ett angivet luftflöde ger ett säkrare värde.

### Uppskattning som sista utväg

Finns varken en passande produkt i Boverket eller ett typvärde gör Aida en egen uppskattning av GWP-fossil A1-A3. Den märks som uppskattning i tabellen.

## Alternativen

### Katalogen

Alternativen kommer ur en katalog med 2 781 miljödeklarationer i 26 kategorier. De flesta kommer från det internationella EPD-systemet Environdec och ungefär var sjätte från norska EPD-Norge. Katalogen är sammanställd i förväg. Produktnamnen i Environdec är oftast på engelska, och en sökning på svenska byggdelsnamn missar det mesta. Några deklarationer i EPD-Norge redovisar A1, A2 och A3 var för sig utan någon summa för A1-A3. Då är A1-A3 summan av de tre, och raden märks med att värdet är summerat. Några deklarationer som inte finns i registren har lästs för hand ur tillverkarens egen EPD (till exempel Kiiltos avjämningsmassor i EPD Hub), med källan i raden.

### Hur alternativen väljs

För varje byggdel får Aida de deklarationer i katalogen som har rätt enhet, sorterade med lägst klimatpåverkan först. Aida väljer två till fyra av dem och motiverar valet mot byggdelens funktion och krav. Aida väljer bara bland deklarationerna i katalogen och hittar aldrig på en produkt. Uttryckta krav, till exempel halkskydd i en entré, är inte förhandlingsbara.

Reglar, balkar och pelare jämförs per löpmeter, konstruktionsskivor per m². Trä och skivor deklareras per m³ och räknas om med tvärsnittet eller tjockleken i komponentens namn. Stål deklareras per kg och räknas om med vikten per meter för en standardprofil som står i namnet: HEA, HEB, IPE och UPE med vikter enligt EN 10365 (ArcelorMittals tabeller), VKR- och KKR-rör med ytterdimension och väggtjocklek (Tibnors konstruktionstabeller, EN 10210 och EN 10219) och tunnplåtsreglar på 45, 70 och 95 mm (Norgips produktkatalog). Vikten och dess källa står i alternativets text, och baslinjen räknas om på samma sätt. Saknas profilen, eller rörets väggtjocklek, görs ingen omräkning, och Aida ber om profilen eller om mängden i kg. Stål jämförs bara inom samma sort: en regel med reglar, en balk med deklarationer för balkar och andra öppna profiler, ett rör med rör. Deklarationer för grovplåt eller tillverkade stålkomponenter, och sådana som inte säger vilken form de gäller, räknas inte om per meter.

Lösa möbler jämförs bara inom samma sort: stolar med stolar, bord med bord, förvaring med förvaring. Förvaring, alltså skåp, hyllor och garderober, räknas som lös inredning om inte namnet säger att den är byggd på plats, till exempel "Platsbyggd garderob", "Inbyggda skåp", "Fast monterade hyllor" eller "Väggmonterade skåp". Då räknas den som fast inredning och jämförs med den. Det är namnet som avgör, så ett förvaringsskåp som kategoriserats som fast inredning jämförs ändå med annan förvaring och inte med köksskåp och bänkskivor.

Mattor och gardiner är två egna sorter och jämförs per m². En matta eller gardin i styck räknas om med storleken i namnet: "Matta 2x3 m" är 6 m² per styck, "Gardin 140x250 cm" 3,5 m². Storleken ska stå med enhet. Saknas den frågar Aida efter den i stället för att anta en typisk storlek. För en gardin är storleken gardinens egen bredd och höjd, inte fönstrets, eftersom en rynkad gardin har mer tyg än fönstret är stort.

Det finns inga miljödeklarationer för lösa mattor i Environdec eller EPD-Norge. Mattor jämförs därför med textilmattor som säljs i rulle (vävda och tuftade, även Axminster och Wilton), per m². Det är samma uppbyggnad som en lös matta: lugg i en bottenväv med en baksida, och tillverkare som Ege säljer mattor ur samma kollektioner som sina rullvaror. Textilplattor räknas inte med, eftersom deras styva baksida av bitumen, PVC eller tjock filt är gjord för att ligga som golv och står för en stor del av vikten. Gardiner jämförs med gardintyger per m² där tillverkaren själv säljer tyget som gardin: i dag 14 deklarationer från Ludvig Svensson i Kinna, som bara finns som PDF och har lagts in för hand. Kvadrats deklarationer anges per kg och gäller tyger för både möbler och gardiner utan att skilja dem åt, och Casalegnos anges per löpmeter gardin utan tygets höjd, så de räknas inte med. Mattornas typvärde hålls tillbaka eftersom 37 av 55 deklarationer kommer från Ege, och gardinernas eftersom alla kommer från Svensson.

Prefabricerad betongstomme jämförs inom samma sorts element: håldäck med håldäck, massiva bjälklag och plattbärlag med varandra, balkar och pelare med balkar och pelare. De flesta betongdeklarationer anges per ton. Ett massivt bjälklag räknas om till m² med tjockleken i komponentens namn och 2 500 kg betong per m³, som i Svensk Betongs elementtabell (en massivplatta på 200 mm väger 500 kg/m²). En balk eller pelare räknas om till löpmeter på samma sätt, med tvärsnittet i namnet. Ett håldäck räknas aldrig om med tjockleken, eftersom hålen skiljer sig mellan tillverkare: enligt Svensk Betong väger ett 200 mm håldäck 255 till 330 kg/m². Ett håldäck per m² jämförs därför bara med håldäck av samma tjocklek vars deklaration själv anger tjocklek och vikt per m², eller med vikten per m² som står i komponentens namn, till exempel leverantörens uppgift. Platsgjuten betong, armering och lättklinker jämförs inte med prefabricerade element. Typvärdet för betongstomme per kg kräver minst 15 deklarationer.

Ett alternativ kan ha högre klimatpåverkan än baslinjen när det är funktionellt relevant, eftersom det som räknas är projektets totala påverkan. Jämförelsen visar då skillnaden med plus eller minus.

Är skillnaden liten mellan två alternativ väljs det med nordisk leverantör, och minst ett alternativ ska ha nordisk leverantör om katalogen har något sådant. Nordisk leverantör avgörs av vem som står bakom deklarationen, inte av deklarationens geografiska giltighetsområde.

När katalogen har jämförbara deklarationer men ingen av dem blir kvar som alternativ, säger raden varför: hur många det var, och om de låg över baslinjen (med siffrorna), var en del av en byggdel snarare än en hel byggdel, eller inte kom med i analysen. "Inga alternativ" står bara när katalogen inte hade något att jämföra med.

### Återbruk

Återbrukade produkter hämtas från Palats, Karlstads kommuns återbruksplattform, med upp till fem annonser per byggdel. Deras klimatpåverkan är en schablon per enhet för transport och enklare renovering, till exempel 0,5 kg CO2e per m² golv och 10 kg CO2e per fönster. Priset är annonsens pris.

Återbrukad stomme (reglar, balkar, skivor och betongelement) har ingen schablon. Där räknas klimatpåverkan som transporten till bygget: komponentens vikt gånger Boverkets värde för transport (modul A4) för samma material. Vikten räknas på samma sätt som baslinjen, ur tvärsnittet eller tjockleken i namnet och Boverkets densitet för trä och skivor, ur standardprofilens vikt per meter för stål och ur 2 500 kg/m³ för massiv betong. En regel 45x95 väger 1,9 kg per löpmeter och får 0,04 kg CO2e per löpmeter som återbrukad, mot 0,12 som ny. Boverkets transportvärde gäller en ny produkt från fabriken via lager till byggplatsen. Sträckan för en viss annons är okänd, och nedmontering och upprustning räknas inte med. Går vikten inte att få fram, till exempel för en regel utan tvärsnitt eller ett håldäck utan leverantörens vikt per m², visas annonsen utan klimatsiffra och med en förklaring.

Varje annons har en länk. Sola byggåterbruks byggmaterial ligger i en öppen butik på Palats som vem som helst kan se. Solas möbler gör det inte: deras annonser går bara att öppna med ett konto på Palats, och då står det "kräver inloggning på Palats" vid länken. Samma sak gäller varje säljare som saknar en öppen butik.

## Priser

Priser tas fram med AI-driven webbsökning mot svenska bygghandlare och entreprenörer. Ett pris avser installerat pris, alltså material och arbete, i kronor exklusive moms. Hittas ett prisintervall används mittpunkten.

Hittas inget pris gör Aida en uppskattning utan källa, och den märks som AI-uppskattat pris. Räcker tiden inte till en uppskattning står posten utan pris hellre än med noll. Varje pris prövas mot ett rimligt intervall för kategorin. Ett pris utanför intervallet märks för kontroll, och ett pris som ligger mer än tre gånger över intervallets högsta värde, eller under en tredjedel av det lägsta, ersätts med intervallets mitt.

Återbruk prissätts med annonsens pris, inte med webbsökning.

## Livslängd

Aida räknar inte med livslängd. Alla klimattal avser produktskedet, en gång, oavsett hur länge produkten håller. Två golv med samma klimatpåverkan per m² ser därför lika bra ut även om det ena håller dubbelt så länge.

Vill du väga in livslängd går det att räkna om till klimatpåverkan per år genom att dela produktskedets värde med livslängden i år. NollCO2-manualen (version 1.2, tabell 2) anger schabloner för förväntad livslängd per byggdel, hämtade ur EU:s ramverk Level(s):

| Byggdel | Förväntad livslängd |
|---|---|
| Grundkonstruktioner och bärverk (BSAB 15.S och 27) | 60 år |
| Inre rumsbildande byggdelar, icke-bärande (BSAB 43) | 30 år |
| Klimatskiljande delar (BSAB 41 och 42) | 30 år, glasfasadelement 35 år, yttre färgskikt 10 år |
| Invändiga ytskikt (BSAB 44) | 10 år |
| Rumskompletteringar (BSAB 46) | 10 år |
| Tappvatten och avlopp (BSAB 52.B och 53.B) | 25 år |
| Värmevatten (BSAB 56.B) | 20 år |
| Luftbehandling, aggregat (BSAB 57) | 20 år |

Schablonen gäller byggdelen och säger inget om en viss produkt. Golvmaterial ligger alla under invändiga ytskikt, så tabellen skiljer inte linoleum från vinyl. Där behövs tillverkarens uppgift om livslängd eller erfarenhet från egna fastigheter, och ett tal som bygger på något annat än en källa ska märkas som uppskattning med sin grund.

## Uppföljning

I läget Uppföljning registrerar du vad som faktiskt byggdes in: vilken produkt, hur mycket och vad det kostade. Utfallet räknas som deklarationens GWP-fossil A1-A3 gånger inbyggd mängd, men bara när enheterna är desamma. Återbruk räknas som noll. Utfall, baslinje och plan jämförs över samma poster, och poster som inte går att räkna namnges.

## Vad som inte räknas

- Transport till byggplatsen och byggproduktionen (A4 och A5).
- Användning, underhåll och utbyte (B1-B7), och därmed livslängd.
- Rivning och avfallshantering (C1-C4) och nyttor bortom livscykeln (D).
- Upptag av biogent kol.
- Transport av återbrukade produkter. Sträckan registreras i uppföljningen men räknas inte om till utsläpp.

## Hur källor och uppskattningar märks

Varje tal ska gå att spåra. I tabellerna märks klimatvärden med sin källa: Boverkets klimatdatabas, EPD-typvärde eller uppskattning. Priser märks som marknadspris eller AI-uppskattat pris.

När Aida svarar på frågor gäller samma princip. Ett tal har helst en källa: Boverkets klimatdatabas, en miljödeklaration, byggriktlinjerna, den här metodbeskrivningen, Palats eller en prisuppgift från webben. Saknas källa ger Aida hellre en uppskattning än inget svar, men då står det att talet är en uppskattning och vad den bygger på.

## Begränsningar

- Resultaten är ett underlag för beslut, inte ett slutgiltigt klimatbokslut.
- Priserna bygger på webbsökning och uppskattningar. Inhämta offerter för exakta värden.
- Baslinjen är en förenkling av NollCO2 och sätts per byggdel, inte för en modellerad typbyggnad.
- AI kan göra fel. Kontrollera källhänvisningarna vid viktiga beslut.
