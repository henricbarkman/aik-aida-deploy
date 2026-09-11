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

Några miljödeklarationer är motsägelsefulla: det redovisade fossilvärdet stämmer inte med deklarationens egen totalsumma. För dem används GWP-GHG i stället, alltså totala växthusgasutsläpp exklusive biogent kol, och rapporten redovisar vilka poster det gäller. GWP-GHG ligger nära GWP-fossil i storlek men är inte samma indikator.

För de flesta kategorier har Aida ett rimligt intervall för klimatpåverkan per enhet, och klimatvärden prövas mot det. Ett värde utanför intervallet märks för kontroll. Ett värde som ligger mer än tre gånger över intervallets högsta värde, eller under en tredjedel av det lägsta, ersätts med intervallets mitt.

## Baslinjen

Baslinjen följer principen i NollCO2, Sweden Green Building Councils metod: den beskriver ett standardfall, alltså vad byggdelen brukar innebära klimatmässigt när den utförs på det sätt som är typiskt i dag. Den är inte ett värsta fall. Aida förenklar metoden genom att sätta ett värde per byggdel i stället för att modellera en hel typbyggnad.

Baslinjen sätts utifrån byggnadstyp och funktion: vad som är konventionellt för en sådan byggdel, inte vad projektet tänker välja. Följde baslinjen projektets eget materialval skulle jämförelsen bli cirkulär.

### Boverkets klimatdatabas först

Aida matchar varje byggdel mot Boverkets klimatdatabas och använder databasens typiska värden (Typical) för A1-A3. En produkt i Boverket används bara när den är samma material som byggdelens standardmaterial, som gipsskiva mot gipsskiva eller betong mot betong. En produkt av annan typ lånas aldrig bara för att den delar basmaterial, till exempel takduk av PVC för ett vinylgolv. Boverket saknar bland annat golvbeläggning, sanitetsporslin, vitvaror och belysning som egna produkter.

### EPD-typvärde när Boverket saknar materialet

Saknar Boverket materialet används ett typvärde ur Aidas EPD-katalog. Typvärdet är medianen av den övre halvan av katalogens värden i kategorin, ungefär den 75:e percentilen. Det motsvarar ett konventionellt val, inte det bästa på marknaden. Ett typvärde kräver minst fem deklarationer, för toaletter fyra.

För golv väljs först en undertyp, till exempel vinyl eller linoleum, utifrån det standardmaterial Aida har antagit. Har undertypen för få deklarationer används hela golvkategorin.

### Uppskattning som sista utväg

Finns varken en passande produkt i Boverket eller ett typvärde gör Aida en egen uppskattning av GWP-fossil A1-A3. Den märks som uppskattning i tabellen.

## Alternativen

### Katalogen

Alternativen kommer ur en katalog med 1 428 miljödeklarationer i 21 kategorier, nästan alla från det internationella EPD-systemet Environdec. Katalogen är sammanställd i förväg. Produktnamnen i Environdec är oftast på engelska, och en sökning på svenska byggdelsnamn missar det mesta.

### Hur alternativen väljs

För varje byggdel får Aida de deklarationer i katalogen som har rätt enhet, sorterade med lägst klimatpåverkan först. Aida väljer två till fyra av dem och motiverar valet mot byggdelens funktion och krav. Aida väljer bara bland deklarationerna i katalogen och hittar aldrig på en produkt. Uttryckta krav, till exempel halkskydd i en entré, är inte förhandlingsbara.

Ett alternativ kan ha högre klimatpåverkan än baslinjen när det är funktionellt relevant, eftersom det som räknas är projektets totala påverkan. Jämförelsen visar då skillnaden med plus eller minus.

Är skillnaden liten mellan två alternativ väljs det med nordisk leverantör, och minst ett alternativ ska ha nordisk leverantör om katalogen har något sådant. Nordisk leverantör avgörs av vem som står bakom deklarationen, inte av deklarationens geografiska giltighetsområde.

### Återbruk

Återbrukade produkter hämtas från Palats, Karlstads kommuns återbruksplattform, med upp till fem annonser per byggdel. Deras klimatpåverkan är en schablon per enhet för transport och enklare renovering, till exempel 0,5 kg CO2e per m² golv och 10 kg CO2e per fönster. Priset är annonsens pris.

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
