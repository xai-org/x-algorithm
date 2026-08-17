use rustc_hash::FxHashSet;
use std::sync::LazyLock;

use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

// Brazil 2026 election filter

// Application providers that use a recommendation system for users must exclude from the
// results the channels and profiles reported to the Electoral Court under the terms of
// § 1º of this article and, except in cases of paid boosting, the content posted on them.

// https://dadosabertos.tse.jus.br/dataset/candidatos-2026

// User ids below are obfuscated; usernames are included for transparency.

// OmarAzizSenador deleted his account at the time this code was written.

/// User ids reported to the Electoral Court for the Brazil 2026 election.
static BRAZIL_2026_ELECTION_USER_IDS: LazyLock<FxHashSet<u64>> = LazyLock::new(|| {
    FxHashSet::from_iter([
        // @renildo
        14160928,
        // @renatoroseno
        14492205,
        // @pedro_lupion
        15022409,
        // @Sen_Cristovam
        20242549,
        // @marcelvanhattem
        21069302,
        // @nilvanferreira
        21571098,
        // @RafaellMilas
        21857669,
        // @chagasvieira
        22035789,
        // @GabrielSouza_RS
        22480147,
        // @Pimenta13Br
        22864100,
        // @rigotto
        23443097,
        // @euserafimcorrea
        24761475,
        // @ManuelaDavila
        25858078,
        // @CintyaMuniz
        26284058,
        // @depguibismarck
        29026134,
        // @cirogomes
        33374761,
        // @aldenorlima
        33973761,
        // @jooliveirapb
        34909888,
        // @caduxavier
        35470350,
        // @NetoAM
        36145775,
        // @marciokieller
        36403221,
        // @AlicePortugal
        36971658,
        // @eniobritodesa
        37654300,
        // @dep_geraldo
        37711911,
        // @inacioarruda
        38659819,
        // @murilogaldinopb
        40040727,
        // @FlavioBolsonaro
        40053694,
        // @BetoAlbuquerque
        40463380,
        // @alexandrekalil
        40721032,
        // @jessicamichels
        41355867,
        // @lcbusato
        42300711,
        // @Alice_Portugal
        42304398,
        // @DepArthurMaia
        42454330,
        // @kruke1
        42487937,
        // @marcelorangel1
        44107208,
        // @perpetua_acre
        44693900,
        // @PedroTaquesMT
        45448098,
        // @betinhogomes
        46958570,
        // @DanielVilela15
        47371488,
        // @zeca_dirceu
        47461491,
        // @eduardopaes
        48298703,
        // @RicardoBarrosPP
        50101324,
        // @danielxdonizet
        50360881,
        // @anyortiz
        50430144,
        // @priscilakrause
        51066167,
        // @advcorrea
        51169114,
        // @afranioboppre50
        51735736,
        // @DeputadoBacelar
        52364013,
        // @JuniorMochi
        52483984,
        // @GuilhermePaz
        52750366,
        // @rosanedopv
        52842057,
        // @RodrigoGuedesam
        52954632,
        // @RollembergPSB
        53050115,
        // @jorginhomello
        53073647,
        // @nelsinhotrad
        54412355,
        // @RubensOtoni
        54539654,
        // @DeputadoWelter
        54545619,
        // @marcosdoval
        54600557,
        // @mineiroptrn
        54897084,
        // @dep_acoutinho
        56480030,
        // @venezianovital
        56734024,
        // @DelegadoMeneses
        57108807,
        // @RobinsonFaria
        57151676,
        // @HarifeViegas
        57163107,
        // @AlineMariano_pe
        58314645,
        // @fredlinhares
        58328715,
        // @JutayMeneses
        59243227,
        // @fernandofilhope
        59323812,
        // @franzepiaui
        61190865,
        // @ale_campelo
        61325857,
        // @carlaayres
        62286707,
        // @HenriqueFontana
        62804559,
        // @BohnGass
        63127680,
        // @JoseAirtonPT
        63868587,
        // @Bobadra
        64302060,
        // @Marco_Brasil
        64434503,
        // @livioluciano
        65059970,
        // @FatimaCleidePT
        65491379,
        // @fabio_novo
        66413485,
        // @miguelcoelhope
        66525428,
        // @VerGuilherme
        66704302,
        // @cantojocelito
        66719730,
        // @betodoisaum
        66753261,
        // @mateus_simoesmg
        66789220,
        // @ciro_nogueira
        67098726,
        // @PattyParente
        67234932,
        // @acirgurgacz
        67601773,
        // @eduardoreiner
        67947379,
        // @andrekamai
        68404402,
        // @costa_rui
        68466700,
        // @MarcosSantanaSC
        68554516,
        // @CarlosBolsonaro
        68712576,
        // @ProfIsrael
        68719944,
        // @juliophilbert
        69073488,
        // @Gyselle_Soares
        70131422,
        // @ernanipolo
        70284501,
        // @deputadoismael
        70453020,
        // @FaissalCalil
        71602909,
        // @SamuelMalafaia
        73442547,
        // @pauderney
        73803827,
        // @geraldoalckmin
        74215006,
        // @eduardobismarck
        74361905,
        // @luizcoutopt
        74738674,
        // @agenorsantospa
        74867163,
        // @santinroveda
        75849275,
        // @fabriciopref
        76039110,
        // @leitaothales
        76224437,
        // @DepMajorAraujo
        76383384,
        // @DepAfonsoHamm
        76741399,
        // @MarcelAlexandre
        77266759,
        // @raquelferreirar
        78707944,
        // @joslene65
        78714361,
        // @profdorinha
        79174387,
        // @capitaotadeu
        80214491,
        // @carlosmatosce
        80597262,
        // @depdarcidematos
        80712669,
        // @Adrianageronim
        80733107,
        // @HermanoMorais
        80995814,
        // @fabiofelixdf
        81384580,
        // @liviaduartepsol
        81742151,
        // @carlosjordy
        82271629,
        // @Manato_es
        82433790,
        // @darypagung
        82936046,
        // @joaoromaneto
        84141432,
        // @andrerochamg
        84196661,
        // @wevertonrocha
        85859074,
        // @realrcoutinho
        86065271,
        // @vozdaenf
        86348991,
        // @romeropelapb
        88303403,
        // @arthurgurgeladv
        90630534,
        // @ANDREAESPANHA
        90898890,
        // @wilsonlimaAM
        91109801,
        // @f_trad
        92509126,
        // @marcelodeputado
        93116173,
        // @tomazteixeira
        93958167,
        // @coroneldavidms
        94378206,
        // @ruycarneiropb
        94428241,
        // @samanthacavalca
        94806323,
        // @requiaooficial
        95253000,
        // @cabogilberto
        95526088,
        // @anapaulalimapt
        95939603,
        // @PROFTULIO
        96570084,
        // @maxmacieldf
        98786988,
        // @michelyfarina
        99617723,
        // @julioarcoverde
        101016613,
        // @PabloValenteDF
        102257530,
        // @NicolasTrancho
        106110720,
        // @DivaneidePT
        107976868,
        // @mayradiasam
        108527180,
        // @HugoMottaPB
        108988113,
        // @deplucasdelima
        109318072,
        // @adjutoafonso
        109332428,
        // @joaopaulodopt
        109657263,
        // @HendersonPinto
        111717686,
        // @alielmachado
        115519533,
        // @luizaogoulart
        116541810,
        // @adelmosoaress
        116751115,
        // @acrisiosena
        117425043,
        // @vereadorjulio
        117594353,
        // @julio_cesar_pi
        119115954,
        // @adrianogaldino
        119586707,
        // @sibellebarros
        120535860,
        // @aryvanazzi
        121422504,
        // @walteralvesrn
        121425571,
        // @netoevangelista
        121594926,
        // @felipecarreras
        122184686,
        // @delegadowaldir
        123689660,
        // @MarceloCastroPI
        125822264,
        // @jardelinacioam
        126323536,
        // @deputadopezenti
        127708617,
        // @ricardozguidi
        127975238,
        // @Mersinho_Lucena
        129681031,
        // @alexandrebaldy
        130620293,
        // @depdelmasso
        132190299,
        // @PedrodoOvo
        132220469,
        // @Larissafgaspar
        132480664,
        // @glaustindafokus
        133692726,
        // @sidneyleite_
        135349877,
        // @LucianoDucci
        136004062,
        // @tadeuveneri
        137919701,
        // @MerlongSolano
        140413929,
        // @lucianogenesio
        142068227,
        // @silvioantonioma
        142501634,
        // @Carloshbfavaro
        143529694,
        // @vicentinhojr
        149013361,
        // @pedrogomatos
        152900822,
        // @dacassia1
        155768056,
        // @profcanguru
        156326262,
        // @AzambujaReinald
        157184730,
        // @MarcosRogerio
        160895960,
        // @MarcoVamosaLuta
        161318399,
        // @marinorpsol
        161404367,
        // @Vanderlan_VC
        162807255,
        // @TJMFernandes
        163251815,
        // @EduardoBraga_AM
        164439493,
        // @crisrbritto
        165329128,
        // @Jeronimoba13
        165754961,
        // @DeputadoRoberio
        165914831,
        // @andrepdt12
        165922194,
        // @geovaniadesa
        165936805,
        // @caiopontess
        165949065,
        // @NelterQueiroz
        166164275,
        // @deputadotoninho
        166417133,
        // @marcocastilhoss
        168408841,
        // @gustavohaguera
        168653875,
        // @TiberioLimeira
        168657354,
        // @walterlfcaval
        170638771,
        // @JanineLucena
        172579844,
        // @tayannystefany
        172581180,
        // @toinharocha2
        173823248,
        // @raquellyra
        174370381,
        // @CinthiaCRibeiro
        175438359,
        // @queirozmfilho
        175901537,
        // @alxlindenmeyer
        179749078,
        // @VictorCoelhoES
        182050588,
        // @rodfvale
        186331377,
        // @MariaSeffair
        199526953,
        // @Anderson_Prego
        201100473,
        // @natbonavides
        203332874,
        // @AntonioCoelhoPe
        208636149,
        // @MeuEuPolitico
        209674215,
        // @saullovianna
        210299199,
        // @fatimacnagitos
        212950503,
        // @celmarcosantos
        219262636,
        // @arilsonchiorato
        220353446,
        // @boscosaraiva
        221399756,
        // @capitaowelton
        222230527,
        // @gardelrolim
        226984087,
        // @osmarfilhoma
        230495542,
        // @luisa_canziani
        230950274,
        // @deltapericles
        239036359,
        // @silascamara_
        243018634,
        // @UrsulaVidalPA
        243429150,
        // @DeAssisDiniz
        247906787,
        // @ranypaulino
        252553750,
        // @rdnarede
        254172269,
        // @deprsantos
        254201392,
        // @wellington_luiz
        255637975,
        // @macielchris
        261472149,
        // @coronelfrota
        263153198,
        // @catulejr
        264391266,
        // @alinegurgel_ap
        267981392,
        // @ayres_jr
        269938072,
        // @alexgalvaodf
        272477039,
        // @diegogarciapr
        273616279,
        // @doutorgutemberg
        274515672,
        // @marcelomaranata
        287876093,
        // @delegado_waldir
        288735634,
        // @coelho_rodrigo
        290204659,
        // @maitebrusman
        294065810,
        // @depkleberRN
        302056921,
        // @WedersonLopes
        313684933,
        // @_sergiosouza
        317973567,
        // @danielvalencapt
        318019551,
        // @depmaracaseiro
        321414691,
        // @loureirocris
        322406139,
        // @jorgeviana
        325131009,
        // @simboramudar22
        337269106,
        // @marisa_lobo
        340331807,
        // @MarceloBelinati
        345512946,
        // @edusantosdf
        348089005,
        // @leilafonsecapb
        349053302,
        // @paulinhoramosap
        349163261,
        // @andrefernm
        360231929,
        // @milkleileite
        365210575,
        // @andresalineiro
        367519089,
        // @Ana_claudiapb
        370942708,
        // @JorgeFrederico2
        398245257,
        // @brena_dianna
        412508127,
        // @faf_freitas
        412669574,
        // @zeliomota
        420614002,
        // @vitordeangelo
        424455116,
        // @instrutormarcio
        437397340,
        // @AllysonBezerra_
        444925483,
        // @CarlosMoises
        456795842,
        // @JulioCesarRib
        459681629,
        // @chicorafa_s
        487108193,
        // @pedroluislongo
        487622592,
        // @MoisesSantosAc
        538269241,
        // @JulianaBRedivo
        545696318,
        // @alcyvania
        556852639,
        // @OthelinoNeto
        583377940,
        // @AfonsoFlorence
        599427558,
        // @elmanooficial
        626602522,
        // @fabiogov55
        630758039,
        // @helenaduailibe_
        632737286,
        // @mitchellemeira
        746090918,
        // @suelenmarques06
        796882578,
        // @JenirNeves
        813802178,
        // @carlaopelobem
        893975196,
        // @Isoldadantaspt
        999393290,
        // @SHEILAKLENER
        1038802238,
        // @rodrigostm10
        1038849745,
        // @xambinhoes
        1068257582,
        // @tatianehelena81
        1075326786,
        // @depjanetedesa
        1094959356,
        // @prof_juniorgeo
        1226451780,
        // @marcoswesleymw
        1308817182,
        // @alexceoficial
        1314840228,
        // @D_GoretePereira
        1362596354,
        // @mickasevalho
        1461892208,
        // @deyvidbacelar
        1526183576,
        // @NegrahLima
        1571332381,
        // @marciopachecopf
        1613063005,
        // @GilvanMaximoOfc
        1651852124,
        // @Brandaveneno
        1710385291,
        // @dr_furlan
        1844887189,
        // @valmirdesergipe
        1960681207,
        // @helinhocastro
        1977089148,
        // @DepAntoniaLucia
        2161527577,
        // @LUANARUIZSILVA
        2216639570,
        // @DepFederalMoses
        2217650233,
        // @BrunoCarianha
        2299610603,
        // @DavidAlmeidaAM
        2353403137,
        // @capitaoassis10
        2359974485,
        // @Francadf_
        2445938091,
        // @Alfredoficial22
        2474258532,
        // @yuriarrudam
        2495584641,
        // @prof_rosaneide
        2523520542,
        // @viagensdaiw
        2572998767,
        // @anadogasoficial
        2583179096,
        // @barbosinhams
        2612421128,
        // @EderMauroPA
        2632802395,
        // @paulo_litro
        2647646724,
        // @LuizianneLinsCE
        2666819714,
        // @LulaOficial
        2670726740,
        // @marcoaurelioITZ
        2691147288,
        // @LucasVergilioGO
        2717062461,
        // @netto_expedito
        2717313965,
        // @tiaomedeiros
        2732504341,
        // @mqueiroga22
        2823869072,
        // @AlbertoMaiaf
        2834745257,
        // @paulolemosap
        2881293743,
        // @manascimentogo
        2892653500,
        // @LizianyM
        2894163148,
        // @carlosveraspt
        3010698441,
        // @mendanhagustavo
        3018780587,
        // @karlosbernardoA
        3029604164,
        // @zuleidequeirozf
        3130842411,
        // @meire_cruvinel
        3205786257,
        // @Marcio_Honaiser
        3294107902,
        // @deputadopriante
        3305760711,
        // @moisesbrazpt
        3373574517,
        // @kleybe_morais
        3512854216,
        // @carmelonetobr
        3662771592,
        // @paulo_mavignier
        4043244676,
        // @zeaugustonalin
        4056653428,
        // @_AlessandroSE
        4250596815,
        // @CarolDeToni
        4566967516,
        // @FofaBorges
        4775264669,
        // @AndreaOficial55
        703604819538407424,
        // @MARTACLERIALIMA
        713343399898820608,
        // @LiaGomesCE
        724003686863740933,
        // @ViniciusFerroC
        726243481669242880,
        // @brisabracchi13
        728292397130592256,
        // @cabo_senna
        744609688415789056,
        // @jarir_pereira
        746804180430487554,
        // @angelaamin11
        766624608338477056,
        // @vanessapstubh
        770298023582765057,
        // @Jorgepinheiroof
        807903184576532480,
        // @amorimvivian_
        820867290749161472,
        // @ProfessorEuler
        824285052590845955,
        // @angelocoronel_
        829391158208032768,
        // @AdelitaMonteiro
        830144764360192002,
        // @ranallipf
        832322309436403712,
        // @GeneralGirao
        841700087143288832,
        // @WillaceSouza
        850775334362505216,
        // @vivitobiasms
        852960064159842305,
        // @rodrigodiasbsb
        863812402680397826,
        // @pedroalmeidace
        882539177757298689,
        // @meuamigojoao
        899602419302240257,
        // @todandara
        904842960931565568,
        // @AlcyPinheiroCE
        909049597091356673,
        // @TorminCassiana
        931087564064329728,
        // @rigoni_felipe
        939423395376136192,
        // @sergiokruke
        940210949943910401,
        // @WilliamSiriRJ
        958357145761845248,
        // @dep_paulinha
        973982865997385728,
        // @Diegoferdfc
        977195483507707904,
        // @ZeCocaOficial
        978602905690427392,
        // @Renatoafjr
        982217430297554945,
        // @fernandomaximoX
        983360917227364353,
        // @RenilceOficial
        984221723544444928,
        // @eng_angelo44
        984522534623301632,
        // @IndiaraNOVO
        986396254287646721,
        // @leaolais_
        986659343436255232,
        // @ptmatiaspedro
        986903953014128641,
        // @KerexuOficial
        992801222926196736,
        // @kekabagno
        993288307625943040,
        // @annakarinapsol
        1009608950176796677,
        // @CapitaoContar
        1011667518728089602,
        // @Trom_Petista
        1013494732834639874,
        // @limmapiaui
        1014568888078651392,
        // @acaroldartora
        1016697971843502080,
        // @RomeuZema
        1020798087449776128,
        // @EduGiraoOficial
        1024403315164160000,
        // @Laurez_Moreira
        1025352673292378113,
        // @zenaidern
        1025870063461720064,
        // @BocaAbertaOf
        1031914738933018624,
        // @RafagninLuciana
        1034079788162535424,
        // @erikaamorimce
        1034138653147242496,
        // @AmauriRibeiroGO
        1036681658760658945,
        // @RenanSantosMBL
        1042601099566436352,
        // @raphaelbarrabr
        1049870508643233793,
        // @pluviapt
        1062505159824150530,
        // @capitaocarpe
        1062678892216020992,
        // @clarissatercio
        1065379080084865024,
        // @capalbertoneto
        1076135802969772032,
        // @tarcisiogdf
        1078618844007157761,
        // @MarcosA_Sampaio
        1080911369447399430,
        // @capitao_alden
        1081245039920144384,
        // @Khalill_gui
        1083098735675084800,
        // @wilkerbarretoam
        1085537170276958208,
        // @PlinioValerio45
        1091691201844207617,
        // @doriel_barros
        1092446595889745921,
        // @joaoluizam
        1092549114762539009,
        // @betopereirams
        1092816222012534784,
        // @biologiagabriel
        1092879030465032192,
        // @ZequinhaMarinho
        1093223195258298368,
        // @DepGuiLandim
        1095422600854032390,
        // @MondardoGiovana
        1095645491419852800,
        // @JuniorManoDep
        1095696286974652422,
        // @IdilvanAlencar
        1095754303367720960,
        // @depsargentolima
        1095990374550695938,
        // @DFDanielFreitas
        1097498693719199744,
        // @CelAlfredo
        1098328459825287187,
        // @benesleocadiorn
        1099003656064643073,
        // @depPedroLucasF
        1104120809482866688,
        // @PollonMarcos
        1105081277135417345,
        // @CoelhoAntonioPE
        1106161642486796288,
        // @SargentoBetania
        1107097837194694663,
        // @JoaoCampos
        1112804630453501953,
        // @SF_Moro
        1113094855281008641,
        // @faustojr_am
        1116362076459601920,
        // @10ronaldomartin
        1118164353121968129,
        // @deputadaniella
        1118481835376431104,
        // @flavionpi
        1118489062866849793,
        // @henriquecesarr
        1121117485900730368,
        // @georgebastos30
        1141394024860966912,
        // @CarboniRogerio
        1143607219302387712,
        // @pedrorochafilho
        1151964529896644620,
        // @washingtonban
        1154733508088270849,
        // @karlasarney_
        1160505413697253376,
        // @jofariasm_
        1167172974451073024,
        // @RobertoCidadeAm
        1168549394830020608,
        // @jadearomero
        1171411853282557952,
        // @prjuniortercio
        1171959927771975680,
        // @FranciscodoPT13
        1188191474636206082,
        // @SandraLimadeVa1
        1199740816018853890,
        // @LeoSuricate
        1208544960032727040,
        // @ManuVieiraSC
        1210296676520513536,
        // @CruzOrleans
        1211689081861623808,
        // @OficialNenemAl
        1216746154626625536,
        // @JairSoutoAM
        1218904655591243777,
        // @JohnRobertPA
        1219003418347483136,
        // @DrVictorAmoras
        1224505061558161408,
        // @KlesleyGarcia
        1224615230195609601,
        // @profterezinhaPT
        1225158156545904642,
        // @brenogaribalde
        1225774842735202306,
        // @gerlanebaccarin
        1225861584972664833,
        // @deputadoelizeu
        1226865113635926021,
        // @dramayraoficial
        1226874382938787841,
        // @BabaTupinamba
        1232082766071767045,
        // @MassamiMiki
        1234278834519920642,
        // @juizcubas
        1234488841479884801,
        // @JacqueMoraes_es
        1235371041712726022,
        // @julinhodeputado
        1238202050745446401,
        // @eusouamom
        1238868377675980803,
        // @fabiolopes_38
        1241821464111853570,
        // @elianexunakalo
        1241878280841674752,
        // @samaramartinsup
        1244783574676627460,
        // @drluizovando
        1247287375803355136,
        // @CoronelFernand9
        1249769581725573121,
        // @jaziel_dr
        1251558672821600259,
        // @cirineu_costa
        1251676296842805257,
        // @fcordeirosc
        1252561439686037506,
        // @fabio_schiochet
        1253698500401008641,
        // @tyago_hoffmann
        1254019417148661760,
        // @peritatirotti
        1254073103661088768,
        // @tanielmacedo
        1254084527422689282,
        // @beckhausersc
        1257020913893195779,
        // @DraSilvana2
        1257476064219140097,
        // @najuliaribeiro
        1259624231765184514,
        // @adailfilhoam
        1260696151029940225,
        // @camilajarams
        1261423487585005575,
        // @mariadoscamelos
        1267127677468753920,
        // @manupeloES
        1269632201760690176,
        // @juliermesenav
        1271216272148176896,
        // @carladicksonrn
        1273322623380983815,
        // @MucioBotelho
        1275549634757365760,
        // @LucasCaregnato
        1275636866075803648,
        // @victorcarvamt
        1280120303818072064,
        // @JuniorGeraldo_
        1280224335307907077,
        // @KodamaThiago
        1281980054730407936,
        // @eugenialima_pe
        1285282863471001601,
        // @daniportelape
        1285295088525082631,
        // @ericodonovo
        1287396715331452932,
        // @MatheusLaiola
        1292497353166004227,
        // @DepCoronel
        1295725787790942208,
        // @gustavosefer
        1298369792169127939,
        // @dinhodowsley
        1298603065457680385,
        // @luladafonte
        1300279136620085249,
        // @marcelinhoguima
        1303443223663325185,
        // @chaves_hildon
        1304134002547331072,
        // @GuguSeba
        1310253107310460933,
        // @CoronelRomualdo
        1310744586139119618,
        // @EduardoRiedel_
        1310977778574032900,
        // @PabloSilvaLira
        1311620519935045632,
        // @cassymonteiro
        1312841615111843848,
        // @delegadkatarina
        1314575606487670784,
        // @anaportelams
        1319825203313184768,
        // @DeoliAnderson
        1322213572953493505,
        // @bocalomoficial
        1335685798746845190,
        // @rosanadasaude_
        1351505730323558400,
        // @dimasfabianomg
        1355216660068761606,
        // @AnaPimentelmg
        1356434859934298115,
        // @faustinorn01
        1356762322489004032,
        // @RafaelFonteles_
        1358739792163389440,
        // @FelipeAlecrimPE
        1375498458132537348,
        // @celiaxakriaba
        1379025327314329608,
        // @marleipr
        1380366998488645636,
        // @apropriajulia
        1382550118314999812,
        // @PrefeitoCesar
        1383362690584768514,
        // @santanna_cs
        1385208512955957250,
        // @sauloportolivei
        1409385758838906882,
        // @juniorferrari55
        1409990621536919555,
        // @deputadodrhugo
        1413128451154866185,
        // @depclaudiac
        1423660842239791115,
        // @coronelbonates
        1427816945735319554,
        // @robertadahorta
        1434501105241792512,
        // @NFgoes
        1435915606541410305,
        // @ScalcoDarlan
        1437758990172184577,
        // @joelrodriguespi
        1447921024117493761,
        // @atilaliraof
        1448414094441291777,
        // @LuizEduardo_RN
        1452666475844706306,
        // @josecam01577970
        1459681826700673030,
        // @RicardoArrruda
        1462770465068437504,
        // @yurydoparedao
        1478380311532736512,
        // @Anderson_ma123
        1478686290270896128,
        // @firmo_oficial
        1481351815669112840,
        // @depjaqueline
        1488866652037029889,
        // @GayerGus
        1489108473027739654,
        // @eribertomfilho
        1491176697445683205,
        // @RosaRezendeGO
        1491387105913810944,
        // @neyamorimac
        1492921263299379207,
        // @sgtgoncalves22
        1504105286876991489,
        // @SKaripuna
        1506313161921777676,
        // @orleansbrandao_
        1510790014111784967,
        // @DelRodrigoSa
        1512105284558282765,
        // @reillerlopes
        1513329845551390720,
        // @franze_carneiro
        1513903701483790343,
        // @mariaarraespe
        1515390927958908928,
        // @SocorroWaquiim
        1516019212145287168,
        // @DanielleDVale
        1516121393028644869,
        // @luizgastaoce
        1519011181574434816,
        // @drviniciuspi
        1519744542173536259,
        // @coronelamadeu
        1523726346232414208,
        // @SindoleyMorais
        1525834800233336832,
        // @JadyelAlencarr
        1531271393219837956,
        // @eudonaneuma
        1533265972659990528,
        // @melcafariaspb
        1534926600743051266,
        // @rejanepstu
        1539038049937547269,
        // @AnneMarques_AP
        1539992525980815364,
        // @soududaramos
        1542878687120482306,
        // @victorlinhalis
        1543943524189720576,
        // @LuisaCela87
        1545437399446077440,
        // @MissiasDias
        1553167499901943808,
        // @catanhopt
        1555559206635315201,
        // @VitormoreiraCG
        1557086339740405761,
        // @Clecio_Luis
        1560635958994812928,
        // @DeboraMenezes22
        1564101654496133120,
        // @weibetapeba
        1575549757900132379,
        // @Arnaldodeputado
        1578127734362046489,
        // @antoniodoidoofc
        1588344483766321154,
        // @gianninogueira2
        1589336373101723649,
        // @Marisaloboreal
        1589768586096148483,
        // @izaarrudape
        1593200340777803776,
        // @davivalencambl
        1594709250835709957,
        // @WickRyanAM
        1612178912947183620,
        // @deputadairacema
        1612229099551858688,
        // @aveiltonsouza
        1612484459034550273,
        // @ToGomes5
        1613522130368356352,
        // @tadeudesouzaam
        1618970134005121025,
        // @gracinhamaosant
        1624490961236631553,
        // @EdsonKambeba
        1633100825403826177,
        // @AdrianaLeal2507
        1637963109573828608,
        // @adrianalmeidapt
        1638181881127612416,
        // @wistongomess
        1648046380106104833,
        // @drbrunoresende_
        1661373663256408066,
        // @DouglasRuas_RJ
        1663563614954086401,
        // @bemoreiradf
        1666457401816391682,
        // @babatupinamba_
        1691097420166254592,
        // @PauloAssun68233
        1702440269138866176,
        // @hebertcsgyn
        1713936872370532352,
        // @moreiramissao
        1721592893511483392,
        // @eduardowilliamm
        1728271638104358912,
        // @bellaccarmelo
        1743284668764463105,
        // @jotinhapiaui
        1744704300616417280,
        // @andrabianchessi
        1752837451175907328,
        // @Lenesilllva
        1771346241449865216,
        // @jorgemaciel05
        1775273264774045697,
        // @CamillaGonda
        1781485235047174144,
        // @aucilene_a75929
        1784209048671322112,
        // @eliabecamposs
        1784275708606320641,
        // @oescobarpro
        1786921594390016001,
        // @coronellrosses
        1793428861348167680,
        // @gabi_bvnt
        1809421300433317892,
        // @passinhoisa
        1812967972358569984,
        // @SandroOmar48237
        1816185635578978304,
        // @007Douggomes
        1821931832008392704,
        // @cironogueirapi
        1844442355421614089,
        // @DragUrbana
        1847431729415397376,
        // @abreu_de63729
        1855074530613403648,
        // @sandsonmenezes
        1856816774517268480,
        // @depcabomacielam
        1859357204278542336,
        // @Lukaovereador
        1874897810753208320,
        // @oiurecastro
        1878758512639033344,
        // @_marquinhostrad
        1883938805306052608,
        // @bolsonaro__jr
        1884423273490108416,
        // @ameliocayresdep
        1887508211445702656,
        // @FelipeVasquesce
        1889410539639517184,
        // @Rpachecopinho
        1898684990428151808,
        // @viscontioficial
        1898913625697292288,
        // @GersonClaroMS
        1899113980544897024,
        // @glenioseixas
        1909726014202171392,
        // @prof_elson_sc
        1937980838299504644,
        // @jotabrandaoam
        1949664156606459905,
        // @NoronhaPro7153
        1958622002719182848,
        // @RubensAngiolett
        1959013340661035008,
        // @Drfilipecm
        1972344363654037504,
        // @DrManuelMarcos
        1975317845962858501,
        // @CarlosCostaPE10
        1975873882654441472,
        // @marciaabrahaodf
        1985760866109964288,
        // @DaClaudio33805
        1988208205575712772,
        // @vandawitoto
        1988323381792681984,
        // @SamaraMadureira
        1990473255539621889,
        // @profraydf
        2010751427623497728,
        // @brenomacedopi
        2013660202873032707,
        // @robson_cacau
        2015964533689323520,
        // @sicchar1389
        2018445464908042240,
        // @GreguyLoooban
        2027415714953396224,
        // @AraceliLemosOF
        2028870149546373120,
        // @missaogabi
        2036817223558234112,
        // @owilsonmartins
        2043714770591735809,
        // @RitaDamore11
        2047488478372368384,
        // @jaimeverruckms
        2052755712644730887,
        // @ThefiAmancio21
        2054040940885532672,
        // @onenencoelho
        2054729517864771584,
        // @edmartresoitao
        2056559848750276608,
        // @obrasilcomlula
        2058595300873289728,
        // @vivianeluizams
        2059726216119107587,
        // @ProfWiterNaves
        2059790203091341316,
        // @DrCrisVeloso
        2061889166779019264,
        // @DaversonMatos
        2062692988606717952,
        // @HVilelaporGoias
        2069828777517973504,
        // @angelaabukce
        2071949683736354816,
        // @drthalescoelho
        2075213887213797376,
        // @Nayladasilva0
        2078543961333850112,
        // @pituxosergipe
        2080630684989730816,
        // @Irmamabelmelo
        2080632627887734784,
        // @thiagocampeloof
        2081154349267345408,
        // @Profsamuelsiebr
        2084048343609540608,
        // @AuriJuniorr13
        2084644771927076864,
        // @CarmemOliverofc
        2084756136004083712,
        // @nandovianapsol
        2086122224126136320,
    ])
});

pub struct Brazil2026ElectionFilter;

impl Brazil2026ElectionFilter {
    fn is_excluded_author(user_id: u64, followed_user_ids: &FxHashSet<u64>) -> bool {
        BRAZIL_2026_ELECTION_USER_IDS.contains(&user_id) && !followed_user_ids.contains(&user_id)
    }

    fn should_remove(candidate: &PostCandidate, followed_user_ids: &FxHashSet<u64>) -> bool {
        let is_excluded = |user_id: u64| Self::is_excluded_author(user_id, followed_user_ids);
        if is_excluded(candidate.author_id) {
            return true;
        }
        if candidate.retweeted_user_id.is_some_and(is_excluded) {
            return true;
        }
        if candidate.quoted_user_id.is_some_and(is_excluded) {
            return true;
        }
        // Drop replies whose conversation context would surface listed authors
        // (ancestors are returned on the scored post for For You thread UI).
        candidate.ancestor_users.iter().copied().any(is_excluded)
    }
}

impl Filter<ScoredPostsQuery, PostCandidate> for Brazil2026ElectionFilter {
    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let followed_user_ids: FxHashSet<u64> = query
            .user_features
            .followed_user_ids
            .iter()
            .map(|&id| id as u64)
            .collect();

        let (removed, kept): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|candidate| Self::should_remove(candidate, &followed_user_ids));

        FilterResult { kept, removed }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stable sample ids from the set (sorted source list order).
    const SAMPLE_LISTED: [u64; 4] = [14160928, 14492205, 15022409, 20242549];

    fn make_candidate(tweet_id: u64, author_id: u64) -> PostCandidate {
        PostCandidate {
            tweet_id,
            author_id,
            ..Default::default()
        }
    }

    #[test]
    fn keeps_candidates_from_non_listed_authors() {
        let filter = Brazil2026ElectionFilter;
        let candidates = vec![
            make_candidate(1, 1),
            make_candidate(2, 2),
            make_candidate(3, 3),
        ];

        let result = filter.filter(&ScoredPostsQuery::default(), candidates);

        assert_eq!(result.kept.len(), 3);
        assert_eq!(result.removed.len(), 0);
    }

    #[test]
    fn removes_candidates_from_listed_authors() {
        let filter = Brazil2026ElectionFilter;
        let listed = SAMPLE_LISTED[0];
        let candidates = vec![
            make_candidate(1, 1),
            make_candidate(2, listed),
            make_candidate(3, 3),
        ];

        let result = filter.filter(&ScoredPostsQuery::default(), candidates);

        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.kept[0].tweet_id, 1);
        assert_eq!(result.kept[1].tweet_id, 3);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].author_id, listed);
    }

    #[test]
    fn removes_retweets_of_listed_authors() {
        let filter = Brazil2026ElectionFilter;
        let listed = SAMPLE_LISTED[1];
        let mut retweet = make_candidate(10, 999);
        retweet.retweeted_user_id = Some(listed);
        let candidates = vec![make_candidate(1, 1), retweet];

        let result = filter.filter(&ScoredPostsQuery::default(), candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 1);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].tweet_id, 10);
    }

    #[test]
    fn removes_quotes_of_listed_authors() {
        let filter = Brazil2026ElectionFilter;
        let listed = SAMPLE_LISTED[2];
        let mut quote = make_candidate(20, 888);
        quote.quoted_user_id = Some(listed);
        let candidates = vec![make_candidate(1, 1), quote];

        let result = filter.filter(&ScoredPostsQuery::default(), candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 1);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].tweet_id, 20);
    }

    #[test]
    fn removes_replies_with_listed_ancestor_users() {
        let filter = Brazil2026ElectionFilter;
        let listed = SAMPLE_LISTED[3];
        let mut reply = make_candidate(30, 777);
        reply.in_reply_to_tweet_id = Some(29);
        reply.ancestors = vec![29, 28];
        reply.ancestor_users = vec![listed, 42];
        let candidates = vec![make_candidate(1, 1), reply];

        let result = filter.filter(&ScoredPostsQuery::default(), candidates);

        assert_eq!(result.kept.len(), 1);
        assert_eq!(result.kept[0].tweet_id, 1);
        assert_eq!(result.removed.len(), 1);
        assert_eq!(result.removed[0].tweet_id, 30);
    }

    #[test]
    fn keeps_replies_with_only_non_listed_ancestor_users() {
        let filter = Brazil2026ElectionFilter;
        let mut reply = make_candidate(40, 777);
        reply.in_reply_to_tweet_id = Some(39);
        reply.ancestors = vec![39, 38];
        reply.ancestor_users = vec![100, 200];
        let candidates = vec![make_candidate(1, 1), reply];

        let result = filter.filter(&ScoredPostsQuery::default(), candidates);

        assert_eq!(result.kept.len(), 2);
        assert_eq!(result.removed.len(), 0);
    }

    #[test]
    fn empty_candidates_list() {
        let filter = Brazil2026ElectionFilter;
        let result = filter.filter(&ScoredPostsQuery::default(), vec![]);

        assert!(result.kept.is_empty());
        assert!(result.removed.is_empty());
    }

    #[test]
    fn all_candidates_listed() {
        let filter = Brazil2026ElectionFilter;
        let listed_a = SAMPLE_LISTED[0];
        let listed_b = SAMPLE_LISTED[1];
        let candidates = vec![make_candidate(1, listed_a), make_candidate(2, listed_b)];

        let result = filter.filter(&ScoredPostsQuery::default(), candidates);

        assert!(result.kept.is_empty());
        assert_eq!(result.removed.len(), 2);
    }

    #[test]
    fn hardcoded_list_is_non_empty() {
        assert!(!BRAZIL_2026_ELECTION_USER_IDS.is_empty());
        assert_eq!(BRAZIL_2026_ELECTION_USER_IDS.len(), 665);
    }

    #[test]
    fn contains_known_listed_ids() {
        let no_follows = FxHashSet::default();
        assert!(Brazil2026ElectionFilter::is_excluded_author(
            40053694,
            &no_follows
        ));
        assert!(Brazil2026ElectionFilter::is_excluded_author(
            21069302,
            &no_follows
        ));
        assert!(Brazil2026ElectionFilter::is_excluded_author(
            1355216660068761606,
            &no_follows
        ));
        assert!(!Brazil2026ElectionFilter::is_excluded_author(
            0,
            &no_follows
        ));
        assert!(!Brazil2026ElectionFilter::is_excluded_author(
            1,
            &no_follows
        ));
        for id in SAMPLE_LISTED {
            assert!(Brazil2026ElectionFilter::is_excluded_author(
                id,
                &no_follows
            ));
        }
    }
}
