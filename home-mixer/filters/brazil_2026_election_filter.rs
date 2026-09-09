use rustc_hash::FxHashSet;
use std::sync::LazyLock;

use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

// Brazil 2026 election filter

// Art. 28 § 1º-A of Electoral Resolution No. 23.610: Application providers that use a
// recommendation system for users must exclude from the results the channels and
// profiles reported to the Electoral Court under the terms of § 1º of this article
// and, except in cases of paid boosting, the content posted on them.

// https://dadosabertos.tse.jus.br/dataset/candidatos-2026

// User ids below are obfuscated; usernames are included for transparency.

// We believe the account @ABR reported by the candidate is not the candidate's actual account, so we are not currently filtering it.
// We believe the account @ACMNETO_ reported by the candidate is not the candidate's actual account, so we are not currently filtering it.
// We believe the account @ALESILVAOFICIAL reported by the candidate is not the candidate's actual account, so we are not currently filtering it.
// We believe the account @BENICIODIAS reported by the candidate is not the candidate's actual account, so we are not currently filtering it.
// We believe the account @CARLOSSAMPAIO reported by the candidate is not the candidate's actual account, so we are not currently filtering it.
// We believe the account @CARLOSVIANA reported by the candidate is not the candidate's actual account, so we are not currently filtering it.
// We believe the account @CLECIACARVALHO1 reported by the candidate is not the candidate's actual account, so we are not currently filtering it.
// We believe the account @DAYSE reported by the candidate is not the candidate's actual account, so we are not currently filtering it.
// We believe the account @PAULODIMELO reported by the candidate is not the candidate's actual account, so we are not currently filtering it.
// We believe the account @PSTUPE reported by the candidate is not the candidate's actual account, so we are not currently filtering it.
// We believe the account @RICAR reported by the candidate is not the candidate's actual account, so we are not currently filtering it.
// We believe the account @SERGINHOCAXIAS reported by the candidate is not the candidate's actual account, so we are not currently filtering it.
// We believe the account @ZANAAMANDA reported by the candidate is not the candidate's actual account, so we are not currently filtering it.
// @ADRIANASOUSAPIAUI no live account found.
// @AGOLDBACH no live account found.
// @AHELIXO no live account found.
// @ALCEU_ALCEUMOREIRA no live account found.
// @ALEXROSETI no live account found.
// @ALKORAP1 no live account found.
// @ARAFETHNASREDDINE no live account found.
// @BETORICHAOFICIAL no live account found.
// @BRUNOPORTODEALMEIDA no live account found.
// @CAIOZMENDONCA no live account found.
// @CHARLES067277 no live account found.
// @CRISTINAGRAEM no live account found.
// @DANIELBRSOARES no live account found.
// @DANILOBALASOFICIAL no live account found.
// @DANILOTORRES100 no live account found.
// @DELEGADOEGUCHI no live account found.
// @DEMAOLIVEIRA70 no live account found.
// @DEPCELSOSABINO no live account found.
// @DEPLUANAREGIA no live account found.
// @DEPUTADOALTAIRSILVA no live account found.
// @DUARTEJR70 no live account found.
// @DUDUSIVINSKI no live account found.
// @EDSONSANTOSRJ no live account found.
// @EUANGELAGARCIALINKTREE no live account found.
// @EXPEDITOFUCAP no live account found.
// @FADAPSICANALISE no live account found.
// @FEDERALFELICIO no live account found.
// @GERSONBURMANNIV no live account found.
// @GSMA1986 no live account found.
// @IGORRAYARN no live account found.
// @JAIZAMETODIO no live account found.
// @LEOMASCARENHASP no live account found.
// @LUCIANALIPPI30 no live account found.
// @LUCIANAOROZIMBO no live account found.
// @LUIZPAULOCURVELO no live account found.
// @MARCELOSILVACAMPINAS no live account found.
// @MEUCANA669499 no live account found.
// @MIRCOCORONETTI no live account found.
// @NADIAGERHARD no live account found.
// @NETOFEITOSA6891 no live account found.
// @PATRICIACRIZANTO2 no live account found.
// @PAULOMOURAOTO no live account found.
// @PEDROPONCIOBE no live account found.
// @PELUCIO_LUCA no live account found.
// @POLICIALPAULOBASTOS no live account found.
// @PRADOCORONEL no live account found.
// @SUSANNAPFEDERAL no live account found.
// @TIAKEYLAECIA no live account found.
// @TWITTERADRIANAACCORSI no live account found.
// @XIGORPORTO no live account found.
// @_ANDREDOPRADO no live account found.
// @_EDUARDOMANTOAN no live account found.

/// Viewer country code the Electoral Court order applies to.
const BRAZIL_COUNTRY_CODE: &str = "br";

/// User ids reported to the Electoral Court for the Brazil 2026 election.
static BRAZIL_2026_ELECTION_USER_IDS: LazyLock<FxHashSet<u64>> = LazyLock::new(|| {
    FxHashSet::from_iter([
        // @madeleinelacsko
        9179462,
        // @nanacachen
        13258012,
        // @renildo
        14160928,
        // @renatoroseno
        14492205,
        // @pedro_lupion
        15022409,
        // @erasintetica
        15585094,
        // @soninhafrancine
        15768105,
        // @AecioNeves
        15828014,
        // @tatyanavaleria
        15908023,
        // @ClariceChacon
        16524287,
        // @rosilenedf
        16526497,
        // @raulchristiano
        16976843,
        // @Rafael_Parente
        16979783,
        // @RicardoFabrizio
        17084658,
        // @marlonluz
        17244270,
        // @eduardopinheiro
        17525670,
        // @falcaopatos
        18950196,
        // @diegotavares
        19546911,
        // @RubinhoDivi
        19617146,
        // @depmariomotta
        19665407,
        // @depchinaglia
        19723670,
        // @Sen_Cristovam
        20242549,
        // @marcelvanhattem
        21069302,
        // @lgmbrasilia
        21438601,
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
        // @fabriziomeller
        22908605,
        // @rigotto
        23443097,
        // @gustavotutuca
        24055827,
        // @camasao50
        24103292,
        // @radiovaldo
        24588898,
        // @euserafimcorrea
        24761475,
        // @rafangeli
        25164669,
        // @pauloteixeira13
        25562342,
        // @crismonteirosp
        25841252,
        // @ManuelaDavila
        25858078,
        // @tomioyano
        26214420,
        // @CintyaMuniz
        26284058,
        // @rodrigolimamdh
        26498680,
        // @Biango
        26560559,
        // @samiabomfim
        27703690,
        // @profsta
        27883076,
        // @Jfelippeneto
        28177866,
        // @depguibismarck
        29026134,
        // @RicardoFerraco
        29434532,
        // @mendoncafilho
        29803424,
        // @hamiltonassis
        30247132,
        // @MarcoMartinsSP
        30495186,
        // @maragabrilli
        30978177,
        // @gleisi
        31139434,
        // @Miwky
        31401730,
        // @ronaldocaiado
        32101740,
        // @lunazarattini
        32160446,
        // @cirogomes
        33374761,
        // @gabrielmagno_13
        33717372,
        // @ticokuzma
        33759666,
        // @CristinaMel
        33911227,
        // @aldenorlima
        33973761,
        // @_sidney_
        34083807,
        // @SharleneAZ
        34286009,
        // @vimarchese
        34374755,
        // @Peter_Costa
        34430921,
        // @RafaCupertino
        34618485,
        // @BetoRicha
        34665220,
        // @marconiperillo
        34790309,
        // @Donato_PT
        34795040,
        // @jooliveirapb
        34909888,
        // @denilsonsoares
        34955073,
        // @julianapt
        35105584,
        // @LeonelRadde
        35268237,
        // @caduxavier
        35470350,
        // @marcofeliciano
        35805725,
        // @aavasantiago
        35827365,
        // @NetoAM
        36145775,
        // @rfalcao13
        36248739,
        // @marciokieller
        36403221,
        // @ThiagoMassuda
        36694689,
        // @PauloMartins10
        36714942,
        // @marceloverly
        36733456,
        // @RenanCeschin
        36753162,
        // @AlicePortugal
        36971658,
        // @edinhosilva
        37055387,
        // @PompeodeMattos
        37320286,
        // @eniobritodesa
        37654300,
        // @pauloserra_sp
        37700244,
        // @dep_geraldo
        37711911,
        // @SouCrisCaiado
        37745764,
        // @ruialves10_
        37949658,
        // @mauricioscalco
        38212742,
        // @MarcioNakashima
        38585953,
        // @inacioarruda
        38659819,
        // @LucasCalilGo
        38929561,
        // @jandira_feghali
        39116717,
        // @tourinhopedro
        39818820,
        // @murilogaldinopb
        40040727,
        // @FlavioBolsonaro
        40053694,
        // @BetoAlbuquerque
        40463380,
        // @alexandrekalil
        40721032,
        // @RomanelliPR
        41271043,
        // @jessicamichels
        41355867,
        // @nidepp
        41403744,
        // @EGCARREIRA
        41410587,
        // @pauloabiackel
        41590536,
        // @profcassiano
        42106436,
        // @KleberfreireAdv
        42202762,
        // @RaoniMendes
        42284334,
        // @lcbusato
        42300711,
        // @Alice_Portugal
        42304398,
        // @DepArthurMaia
        42454330,
        // @marciofrancasp
        42455446,
        // @kruke1
        42487937,
        // @Domingos_Neto
        42626592,
        // @ArlenSantiago
        42630237,
        // @Francischini_
        42736936,
        // @onyxlorenzoni
        43041690,
        // @pedropaulo
        43189774,
        // @axelgrael
        43326346,
        // @owagnerpro
        43364776,
        // @edurodrigues_25
        43856097,
        // @profleomatos
        44070650,
        // @marcelorangel1
        44107208,
        // @Fufaazevedo
        44153555,
        // @renatasouzario
        44195955,
        // @duarte_nogueira
        44455876,
        // @rafaelcampelo
        44460118,
        // @rafamacris
        44690902,
        // @perpetua_acre
        44693900,
        // @pedrosuplicy30
        44739698,
        // @FelixMendoncaJr
        45235841,
        // @PedroTaquesMT
        45448098,
        // @CarlosZarattini
        45473463,
        // @MarceloFreixo
        45870897,
        // @alicedrummond
        46204905,
        // @adriano_viana
        46229863,
        // @RodrigoSchroder
        46313017,
        // @Igorbaimapsol
        46647000,
        // @SolangeFreitas
        46891928,
        // @betinhogomes
        46958570,
        // @DanielVilela15
        47371488,
        // @RicMarques_RM
        47387995,
        // @chrispuppi
        47418811,
        // @zeca_dirceu
        47461491,
        // @augustocury
        47529845,
        // @policarpodf
        47851105,
        // @Altineu
        47991805,
        // @fernandojordao
        48062025,
        // @marciojerry
        48122425,
        // @eduardopaes
        48298703,
        // @gilbertokassab
        49051293,
        // @senadorcidgomes
        49089646,
        // @Camilo_Capi
        49339134,
        // @paulopinheirorj
        49437070,
        // @nelioaguiarstm
        49597151,
        // @guilhermepasin
        49616087,
        // @cicerolucena
        49632512,
        // @bethsahao
        49790504,
        // @FernandoLadeia
        49818264,
        // @ZimbaldiRafa
        49848446,
        // @acmnetoba
        49966078,
        // @ReginaldoLopes
        50088692,
        // @RicardoBarrosPP
        50101324,
        // @zecadopt
        50133997,
        // @danielxdonizet
        50360881,
        // @anyortiz
        50430144,
        // @ludiocabral
        50514890,
        // @deppauloguedes
        50713061,
        // @priscilakrause
        51066167,
        // @advcorrea
        51169114,
        // @noelnit
        51178338,
        // @VandinhoLeite
        51483562,
        // @VanildaBordieri
        51498674,
        // @afranioboppre50
        51735736,
        // @blogdogarotinho
        51834513,
        // @AlexandreCuri
        51882661,
        // @celinaleao
        51935259,
        // @DarcioVix
        51946581,
        // @jilmartatto
        52045368,
        // @thigagliasso
        52297557,
        // @DeputadoBacelar
        52364013,
        // @amastha2026
        52371426,
        // @JuniorMochi
        52483984,
        // @ADALBERTO_MDB
        52540070,
        // @duiliodecastro
        52564555,
        // @aknoploch
        52690350,
        // @OsmarTerra
        52722451,
        // @lidicedamata
        52724814,
        // @DepTraiano
        52736824,
        // @GuilhermePaz
        52750366,
        // @rosanedopv
        52842057,
        // @RodrigoGuedesam
        52954632,
        // @joliveirambl
        52971495,
        // @RollembergPSB
        53050115,
        // @jorginhomello
        53073647,
        // @neyleprevost
        53163365,
        // @DayseHansa
        53473894,
        // @renanroto
        53544192,
        // @advandrebarros
        53681220,
        // @AngeloAlmeidaBA
        53719510,
        // @paulosalimmaluf
        53776600,
        // @tatiroque
        53911544,
        // @kikosilveira
        53998138,
        // @VINIANZILIERO
        54316589,
        // @nelsinhotrad
        54412355,
        // @RubensOtoni
        54539654,
        // @DeputadoWelter
        54545619,
        // @vanderloubet
        54547924,
        // @xuxudalmolin
        54585047,
        // @marcosdoval
        54600557,
        // @Reimont
        54612905,
        // @flavinhocn
        54735703,
        // @mineiroptrn
        54897084,
        // @luizaerundina
        55022037,
        // @RodLago
        55152459,
        // @simaopedro_SP
        55366483,
        // @silvionavarro
        55366770,
        // @HomeroMarchese
        55370663,
        // @giriboni
        55385927,
        // @RobertoCarlosTB
        55450890,
        // @BiologoHenrique
        55544977,
        // @cristianostpr
        55558067,
        // @stephanesjunior
        55607796,
        // @eduardo_vidal
        55686825,
        // @vitor_bicca
        56081072,
        // @Jrsantosrosa
        56400752,
        // @EduPinheiro7022
        56409767,
        // @MauricioPeixer
        56444778,
        // @Iran_Barbosa
        56466859,
        // @dep_acoutinho
        56480030,
        // @joaonatel
        56610250,
        // @FredCostaDep
        56689975,
        // @venezianovital
        56734024,
        // @marlonreis
        56829466,
        // @romerojuca
        56843641,
        // @rosedefreitas
        56864413,
        // @sandesjunior
        56922936,
        // @fernandapsol
        57044549,
        // @josefortunati
        57106033,
        // @DelegadoMeneses
        57108807,
        // @RobinsonFaria
        57151676,
        // @HarifeViegas
        57163107,
        // @Rafaelpicciani
        57529926,
        // @CanzianiAlex
        57641073,
        // @depHugoLeal
        57771926,
        // @gabriellimambl
        58029195,
        // @KarenGregoris
        58044789,
        // @marinapassadore
        58247896,
        // @AlmeidaMarcus
        58293305,
        // @AlineMariano_pe
        58314645,
        // @fredlinhares
        58328715,
        // @yurimourarj
        58340662,
        // @depluizfernando
        58483633,
        // @Dyamondharper
        58517952,
        // @carloselula
        58538668,
        // @welbert__pedro
        58661156,
        // @GugaJP
        58878180,
        // @LeonoraPerico
        58911503,
        // @adrianofritz
        59135814,
        // @JutayMeneses
        59243227,
        // @fernandofilhope
        59323812,
        // @louiselima_md
        59346999,
        // @CarlosBezerraJr
        59483409,
        // @fernandabarth
        59534429,
        // @brunotopete
        59855833,
        // @subgonzagamg
        59868993,
        // @nandopinheiro22
        60150569,
        // @marcio_motta
        60469505,
        // @anapaulagold
        60708760,
        // @claudioapolina
        60731692,
        // @Isquierdorio
        60805457,
        // @andrerodini
        60879681,
        // @luisbenambl
        60983130,
        // @CovattiFilho
        60994156,
        // @FelicioRamuth
        61119320,
        // @franzepiaui
        61190865,
        // @mariocaixa
        61196446,
        // @SigaPepeVargas
        61208942,
        // @ale_campelo
        61325857,
        // @honoratopvh
        61472552,
        // @pauloabarbosa
        61579803,
        // @carlaayres
        62286707,
        // @juliacasamasso
        62289167,
        // @jaqueswagner
        62501888,
        // @HenriqueFontana
        62804559,
        // @fabiotokarski
        62906951,
        // @mariadorosario
        63118359,
        // @BohnGass
        63127680,
        // @maurotramonte
        63158278,
        // @luizsarraf
        63164895,
        // @FilipeSabara
        63474515,
        // @orlandopesoti
        63494090,
        // @leandrograss
        63507573,
        // @marceloaro
        63510130,
        // @vereadorsamuel1
        63817204,
        // @JoseAirtonPT
        63868587,
        // @Bernarditv
        64297102,
        // @Bobadra
        64302060,
        // @antonionetopdt
        64391577,
        // @Marco_Brasil
        64434503,
        // @wladmesquita
        64460310,
        // @josenunes_ARI
        64482750,
        // @hmecabo
        64493005,
        // @helencabral13
        64493500,
        // @romulorippa
        64605196,
        // @JulioLopesRio
        64755437,
        // @livioluciano
        65059970,
        // @tenentemelo
        65194252,
        // @FatimaCleidePT
        65491379,
        // @Glauber_Braga
        65720380,
        // @ladyfontenelle
        65763684,
        // @marcoslatino
        65972973,
        // @fabio_novo
        66413485,
        // @OgierBuchi
        66459281,
        // @miguelcoelhope
        66525428,
        // @Brigadeiromma
        66529851,
        // @VerGuilherme
        66704302,
        // @cantojocelito
        66719730,
        // @betodoisaum
        66753261,
        // @mateus_simoesmg
        66789220,
        // @iriny_13
        66810575,
        // @carlosgiannazi
        66814653,
        // @Priscilaromano
        66900651,
        // @Assis_Gurgacz_N
        66982240,
        // @monica_benicio
        67002637,
        // @UlissesMaia
        67009496,
        // @anaperugini
        67061352,
        // @ciro_nogueira
        67098726,
        // @LarissaPucca
        67184191,
        // @PattyParente
        67234932,
        // @acirgurgacz
        67601773,
        // @gersongabrielli
        67727566,
        // @gleisonpego
        67737645,
        // @eduardoreiner
        67947379,
        // @celamericosp
        67978929,
        // @lucasmortimer
        68154178,
        // @fabiofeijo
        68181274,
        // @andrekamai
        68404402,
        // @costa_rui
        68466700,
        // @AloisioParana
        68486960,
        // @MarcosSantanaSC
        68554516,
        // @clarianabarao
        68578439,
        // @joaopaulorillo
        68694603,
        // @CarlosBolsonaro
        68712576,
        // @ProfIsrael
        68719944,
        // @DatenaOficial
        68722955,
        // @StelaFarias
        68759158,
        // @DenisAndia
        68763092,
        // @vanbrandao
        68891838,
        // @acrodriguessp
        69016521,
        // @noraldinojunior
        69020170,
        // @juliophilbert
        69073488,
        // @MonicaSeixas
        69370960,
        // @esuplicy
        69373037,
        // @juniorsinforma
        69874322,
        // @Gyselle_Soares
        70131422,
        // @ernanipolo
        70284501,
        // @pedroribeiropdt
        70301797,
        // @janetepieta
        70361625,
        // @lucarminatti
        70405671,
        // @luis_fabuloso
        70443471,
        // @deputadoismael
        70453020,
        // @jonasdonizette_
        70956209,
        // @rdlorenzoni
        71056246,
        // @sen_wellington
        71065638,
        // @charlesribeiro_
        71098452,
        // @lindberghfarias
        71310152,
        // @FabioFayad
        71345547,
        // @Daniel_PCdoB
        71545154,
        // @FaissalCalil
        71602909,
        // @NabilBondukiSP
        71613541,
        // @lohannaf
        71917109,
        // @Maristeladutra
        71953093,
        // @SANDRAFARAJ
        72296368,
        // @maricarvalhoro
        72556466,
        // @aaluisfernando
        72575848,
        // @DepPatriciaAlba
        72949846,
        // @marcoalba
        73034114,
        // @EmidioDeSouza_
        73217377,
        // @pedrolimasjc
        73288051,
        // @SamuelMalafaia
        73442547,
        // @rowennabrito
        73478714,
        // @pauderney
        73803827,
        // @geraldoalckmin
        74215006,
        // @WaldeckCarneiro
        74234605,
        // @AlencarBraga13
        74243131,
        // @eduardobismarck
        74361905,
        // @LeurLomantoJr
        74538721,
        // @RogerPerestelo
        74583511,
        // @Casagrande_ES
        74722174,
        // @luizcoutopt
        74738674,
        // @VICENTINHOPT
        74762633,
        // @MCrivella
        74860967,
        // @agenorsantospa
        74867163,
        // @caetraven
        75035844,
        // @profpaulamarisa
        75058892,
        // @dasilvabenedita
        75060031,
        // @cassioafsoares
        75128422,
        // @kaiofeitosa
        75163315,
        // @santinroveda
        75849275,
        // @Gisele_Nasc
        75977555,
        // @fabriciopref
        76039110,
        // @drwilsonbatista
        76043437,
        // @senadorhumberto
        76049312,
        // @carloschiodini
        76093489,
        // @paulofiorilo
        76206825,
        // @leitaothales
        76224437,
        // @sofiacavedonPT
        76329038,
        // @DepMajorAraujo
        76383384,
        // @guerinocolatina
        76698584,
        // @DepAfonsoHamm
        76741399,
        // @Patrus_Ananias
        77210725,
        // @MarcelAlexandre
        77266759,
        // @malafaia_d
        77297183,
        // @denisespessoa
        77658135,
        // @ClesioSalvaro
        77860393,
        // @rodrivaladares
        78154723,
        // @AlexManente23
        78673777,
        // @leo_picciani
        78690962,
        // @raquelferreirar
        78707944,
        // @joslene65
        78714361,
        // @profdorinha
        79174387,
        // @colattodeputado
        79252957,
        // @gilbertoabramo
        80095058,
        // @michelschlemper
        80123403,
        // @capitaotadeu
        80214491,
        // @LuisTibeOficial
        80307396,
        // @MarcioMacedoPT
        80548664,
        // @aldairrizzi
        80559557,
        // @heldersalomao
        80575628,
        // @aniellefranco
        80582734,
        // @carlosmatosce
        80597262,
        // @EvairdeMelo
        80626542,
        // @depdarcidematos
        80712669,
        // @Adrianageronim
        80733107,
        // @HermanoMorais
        80995814,
        // @fabiofelixdf
        81384580,
        // @marciusmachado
        81445552,
        // @liviaduartepsol
        81742151,
        // @DaviSacer
        81836518,
        // @laurasito
        82143155,
        // @IzalciLucas
        82144764,
        // @carlosjordy
        82271629,
        // @Joyceeduca
        82412969,
        // @CassianoCaron
        82415360,
        // @Manato_es
        82433790,
        // @leonelquerino
        82760526,
        // @DepMauricioRS
        82776436,
        // @darypagung
        82936046,
        // @rodrigocruzz
        83489126,
        // @MUNIQUEBUSSON
        83582018,
        // @uczai
        83722886,
        // @FRANBONI
        83730940,
        // @alberto_fraga
        83844236,
        // @_LuizEmanuel
        84127211,
        // @joaoromaneto
        84141432,
        // @andrerochamg
        84196661,
        // @ProfPicler
        84325242,
        // @MaguinhaMalta
        84396300,
        // @oliviasantana65
        84608790,
        // @filhomarcio
        84639445,
        // @lubloureiro
        85150664,
        // @RobertoPSOL
        85293890,
        // @randolfeap
        85327394,
        // @emanuelcacho
        85461555,
        // @danielter
        85585608,
        // @maxlemos
        85613796,
        // @PCBpartidao
        85647830,
        // @wevertonrocha
        85859074,
        // @realrcoutinho
        86065271,
        // @vozdaenf
        86348991,
        // @Baleia_Rossi
        86373674,
        // @JeanVolpato
        86825438,
        // @DomingosSavioMG
        87473436,
        // @romeropelapb
        88303403,
        // @Doutorluizinhot
        89419656,
        // @DepJeferson
        89787624,
        // @luizinhopatria
        90083628,
        // @carladuwe
        90213796,
        // @MariliaArraes
        90309661,
        // @MGRodrigoCastro
        90478644,
        // @arthurgurgeladv
        90630534,
        // @FernandoCFAR
        90687549,
        // @ANDREAESPANHA
        90898890,
        // @wilsonlimaAM
        91109801,
        // @RaffieDellon
        91840828,
        // @f_trad
        92509126,
        // @felipeaugusto01
        92528438,
        // @LucasRedecker
        92739280,
        // @dpmarciomarinho
        92743190,
        // @nancythame
        93065930,
        // @marcelodeputado
        93116173,
        // @bupessoa
        93424680,
        // @danielsoranz
        93426187,
        // @DuarteBechir
        93894534,
        // @tomazteixeira
        93958167,
        // @DrRafaelFavatto
        94115118,
        // @Obsevador
        94167900,
        // @coroneldavidms
        94378206,
        // @ruycarneiropb
        94428241,
        // @netemoura
        94450303,
        // @samanthacavalca
        94806323,
        // @EdicarlosVieira
        94943878,
        // @requiaooficial
        95253000,
        // @cabogilberto
        95526088,
        // @overissimo
        95621286,
        // @deciolimasc
        95699837,
        // @anapaulalimapt
        95939603,
        // @PROFTULIO
        96570084,
        // @cassioandradepa
        96991211,
        // @TrzeciakDaniel
        97587760,
        // @evagoncalvesmg
        97677980,
        // @maxmacieldf
        98786988,
        // @matheusquintal
        98892109,
        // @alexandrenau
        99001454,
        // @ailtonlopespsol
        99389654,
        // @michelyfarina
        99617723,
        // @CacaLeao
        100521721,
        // @julioarcoverde
        101016613,
        // @fernando_it
        101892949,
        // @PabloValenteDF
        102257530,
        // @EduardoGomesTO
        102444270,
        // @falcon
        103423503,
        // @taliriapetrone
        103704608,
        // @eubrasileiro_
        103723639,
        // @RafaelGreca
        104819806,
        // @MarinaSilva
        105155795,
        // @gustavopetta
        105236036,
        // @nubiapassos
        105799989,
        // @AmaliaTortato
        105872129,
        // @NicolasTrancho
        106110720,
        // @ThabattaPimenta
        106126597,
        // @brendamars
        107012654,
        // @DivaneidePT
        107976868,
        // @tenenteromulo
        108322618,
        // @mayradiasam
        108527180,
        // @gui_pugliese
        108937956,
        // @HugoMottaPB
        108988113,
        // @rosenvergreis
        109006854,
        // @DiogoForjaz
        109147422,
        // @deplucasdelima
        109318072,
        // @adjutoafonso
        109332428,
        // @joaopaulodopt
        109657263,
        // @paulaschild
        110187886,
        // @depchicoalencar
        110522807,
        // @thomeprefeito
        110560697,
        // @Cleiton1Pereira
        110876570,
        // @jeanwyllys_real
        111123176,
        // @AndreMouraSE
        111148190,
        // @talitagalhardo
        111695906,
        // @HendersonPinto
        111717686,
        // @marialuciaamary
        112476794,
        // @dep_padrejoao
        113885149,
        // @alielmachado
        115519533,
        // @Silvio_CFilho
        115657305,
        // @deputadomarcon
        115676753,
        // @luizaogoulart
        116541810,
        // @mahomsi
        116736901,
        // @adelmosoaress
        116751115,
        // @EdegarPretto
        116829646,
        // @acrisiosena
        117425043,
        // @fredprocopioofc
        117490508,
        // @vereadorjulio
        117594353,
        // @ronaldornrn
        117801614,
        // @patybenzaquem
        118492432,
        // @ZeRicardoAM
        119079224,
        // @julio_cesar_pi
        119115954,
        // @capitaosamuelof
        119416561,
        // @adrianogaldino
        119586707,
        // @AndreQuintaoPT
        119818761,
        // @sibellebarros
        120535860,
        // @aryvanazzi
        121422504,
        // @walteralvesrn
        121425571,
        // @netoevangelista
        121594926,
        // @NelsonHossri
        121954870,
        // @depulyssesgomes
        122123728,
        // @felipecarreras
        122184686,
        // @delegadowaldir
        123689660,
        // @Valdeci13RS
        124157459,
        // @FlavioCoutinho_
        124360967,
        // @Thais_Margarido
        124531510,
        // @drgeorgemorais
        124753094,
        // @edmundosouza7
        124879050,
        // @OlenoMatos
        125433649,
        // @dario4e20
        125727969,
        // @annemouraam
        125816456,
        // @MarceloCastroPI
        125822264,
        // @susanevidal
        125851531,
        // @anaffonso13
        126091135,
        // @jardelinacioam
        126323536,
        // @deputadopezenti
        127708617,
        // @lucasneves_sc
        127952305,
        // @ricardozguidi
        127975238,
        // @ortizjuniorm
        128550627,
        // @GodriJunior
        128906180,
        // @apjunqueira
        129055364,
        // @ninamarinabraga
        129070751,
        // @Mersinho_Lucena
        129681031,
        // @depleomonteiro
        129837652,
        // @alexandrebaldy
        130620293,
        // @paulaodopt
        131004593,
        // @minc_rj
        131286287,
        // @Miriampetrone
        132187525,
        // @depdelmasso
        132190299,
        // @PedrodoOvo
        132220469,
        // @Larissafgaspar
        132480664,
        // @betofantinel
        132735215,
        // @40rodrigofarias
        133685785,
        // @glaustindafokus
        133692726,
        // @drthiagopeixoto
        134284069,
        // @sidneyleite_
        135349877,
        // @LucianoDucci
        136004062,
        // @matheuscadorin
        136381913,
        // @paulomansur_
        136710714,
        // @andreia_zito
        137369968,
        // @BiradoPindare
        137548563,
        // @tadeuveneri
        137919701,
        // @sigageraldoluis
        138172915,
        // @Eduardo_Cury
        138504441,
        // @Katiadiasjf
        139059131,
        // @rodrigominotto
        139809677,
        // @MerlongSolano
        140413929,
        // @raphaelsebba
        140485170,
        // @wiliantonezi
        141023529,
        // @DiogoPBotelho
        141087783,
        // @lucianogenesio
        142068227,
        // @gilmarribeirojr
        142309506,
        // @silvioantonioma
        142501634,
        // @moemagramacho
        142707910,
        // @NelsiWelter
        142895907,
        // @Carloshbfavaro
        143529694,
        // @joaorodriguessc
        143924396,
        // @SenadorRogerio
        144372753,
        // @oacelio
        144599667,
        // @fredpachecorj
        144681516,
        // @majorpalumbo
        147744504,
        // @dep_fatimanunes
        148355994,
        // @depzanchin
        148424655,
        // @vicentinhojr
        149013361,
        // @sanchilispe
        149518311,
        // @marcospereira04
        149746462,
        // @YvisEvelynn
        150019112,
        // @narciakelly
        150448458,
        // @DrLindoso
        151034428,
        // @helderbarbalho
        151653693,
        // @pedrogomatos
        152900822,
        // @CezinhaNunes
        153131779,
        // @AntonioTelesJr
        153268918,
        // @RicardoCappelli
        153563550,
        // @JarbasFilho_
        155230905,
        // @FabiolaMansur_
        155564771,
        // @dacassia1
        155768056,
        // @sgtalexandre
        156144539,
        // @profcanguru
        156326262,
        // @Sergio_Turra
        156355487,
        // @AzambujaReinald
        157184730,
        // @JoseMedeirosMT
        157226645,
        // @RenataBuenoITA
        158067709,
        // @Marivaldo4P
        159643822,
        // @CoronelMedina14
        160541377,
        // @sandroalexpr
        160554168,
        // @MarcosRogerio
        160895960,
        // @MarcoVamosaLuta
        161318399,
        // @marinorpsol
        161404367,
        // @LianaCirne
        161416659,
        // @maneco_hassen
        162794224,
        // @Vanderlan_VC
        162807255,
        // @TJMFernandes
        163251815,
        // @emidinhomadeira
        163546697,
        // @MariliaPFerrari
        163606193,
        // @Ruicpimenta29
        163831440,
        // @DepZeMilton
        163935204,
        // @wandnogueira
        164012285,
        // @dimasgadelha13
        164058788,
        // @DepNeilando
        164416842,
        // @EduardoBraga_AM
        164439493,
        // @profjosemarpsol
        164478764,
        // @TiagoCado
        164816642,
        // @crisrbritto
        165329128,
        // @_Heloisa_Helena
        165499618,
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
        // @rafacastro_40
        166612108,
        // @artagaojunior
        166645637,
        // @depsamueljunior
        167761418,
        // @serafinipsol
        168155564,
        // @WladGarotinho
        168360473,
        // @marcocastilhoss
        168408841,
        // @drthiagoduarte
        168504384,
        // @liberato1125
        168510620,
        // @RachelSherazade
        168520768,
        // @gustavohaguera
        168653875,
        // @TiberioLimeira
        168657354,
        // @edbernasil
        168814204,
        // @pepecollaco
        169158984,
        // @DepEzequielRN
        169180911,
        // @matheusmanholer
        170176086,
        // @pedrofrancez
        170188044,
        // @depJeferson10
        170505916,
        // @walterlfcaval
        170638771,
        // @wilsonsousajr
        171158811,
        // @giovaniculau
        171278562,
        // @dudualfaia
        172117779,
        // @JanineLucena
        172579844,
        // @tayannystefany
        172581180,
        // @rdrgcassollima
        172883136,
        // @arthurboavista
        173527362,
        // @toinharocha2
        173823248,
        // @rodrigomoraes65
        174272995,
        // @raquellyra
        174370381,
        // @deputadoalisson
        174507960,
        // @CinthiaCRibeiro
        175438359,
        // @ThiagoBagatin
        175898388,
        // @queirozmfilho
        175901537,
        // @julianecviana
        176245917,
        // @RenanFilho_
        176502945,
        // @MiltonVieiraOfc
        176920334,
        // @bordalopt
        177595081,
        // @alxlindenmeyer
        179749078,
        // @leatricebez
        179938194,
        // @JulyverModesto
        180256758,
        // @VictorCoelhoES
        182050588,
        // @eupugina
        184098414,
        // @BrunoEnglerDM
        184210354,
        // @rogeriobarrapa
        185053556,
        // @FaveroNeto
        185482131,
        // @rodfvale
        186331377,
        // @Andiara1400
        187726597,
        // @acarlosmendes
        190308328,
        // @pablomarcal
        191223319,
        // @lorran_rebeldia
        192010152,
        // @esperidiaoamin_
        192658045,
        // @pretagilsa
        193045733,
        // @joeldaharpa
        195423732,
        // @MarciaTaschetti
        198338329,
        // @JacksonAndre7
        198350574,
        // @DepEduardoCunha
        198535390,
        // @EdsonSilvaCotia
        199102127,
        // @MariaSeffair
        199526953,
        // @Anderson_Prego
        201100473,
        // @silvyealves
        202856731,
        // @natbonavides
        203332874,
        // @reisptsp
        203784995,
        // @profagraciele
        203948587,
        // @profhenriquejr
        204042807,
        // @Rafaeik
        204136918,
        // @toninhobondade
        204759112,
        // @lucasrfamaral
        205959233,
        // @luboiteux
        207739610,
        // @AntonioCoelhoPe
        208636149,
        // @ivaniserotta
        209071918,
        // @MeuEuPolitico
        209674215,
        // @saullovianna
        210299199,
        // @fatimacnagitos
        212950503,
        // @Odarlone
        213105137,
        // @betaopt
        213382232,
        // @rafaelprimo
        215727649,
        // @Lucinildo
        217347443,
        // @celmarcosantos
        219262636,
        // @LaizPerrut
        220062558,
        // @arilsonchiorato
        220353446,
        // @boscosaraiva
        221399756,
        // @capitaowelton
        222230527,
        // @DATENAREAL
        223852535,
        // @LuisMirandaUSA
        225426734,
        // @filipebarrost
        225925013,
        // @gardelrolim
        226984087,
        // @emersonbacil
        229028677,
        // @gislenemoura
        229229129,
        // @osmarfilhoma
        230495542,
        // @marisol_santoss
        230769498,
        // @luisa_canziani
        230950274,
        // @caroldecaxias
        231379712,
        // @kelitalks
        235765762,
        // @euberlucas
        237071521,
        // @lyneoliver
        237497430,
        // @elzefacchinetti
        237562432,
        // @wildermorais
        237687306,
        // @deltapericles
        239036359,
        // @_lucavalcante
        239059356,
        // @silascamara_
        243018634,
        // @UrsulaVidalPA
        243429150,
        // @gutoschiavetto
        243429651,
        // @moarasaboia
        244491558,
        // @depgurgel
        245320150,
        // @RogerioCorreia_
        245392082,
        // @gelizabetesp
        247745869,
        // @DeAssisDiniz
        247906787,
        // @iginomarcos13
        250070980,
        // @prgilsondesouza
        250214822,
        // @ranypaulino
        252553750,
        // @RachelMaroja
        253219200,
        // @rdnarede
        254172269,
        // @deprsantos
        254201392,
        // @SargentoFAHUR
        255300173,
        // @wellington_luiz
        255637975,
        // @fredantunes_rs
        256499222,
        // @DelegadoJacovos
        257247983,
        // @TamyresFilgueir
        258957259,
        // @Rodrigopreis13
        259022842,
        // @Marceloalvaroan
        261007129,
        // @Nenebabyvianna
        261446056,
        // @macielchris
        261472149,
        // @paulobregolin
        262375578,
        // @coronelfrota
        263153198,
        // @victorantoun
        264298649,
        // @catulejr
        264391266,
        // @deboratruck7
        266856351,
        // @AureoRibeiroRJ
        267467458,
        // @alinegurgel_ap
        267981392,
        // @depsebastiao
        268318846,
        // @pastorflamarion
        269618056,
        // @ayres_jr
        269938072,
        // @RafaelMottaRN
        271212223,
        // @alexgalvaodf
        272477039,
        // @diegogarciapr
        273616279,
        // @JairMiotto
        273974254,
        // @doutorgutemberg
        274515672,
        // @BiaCerqueira_
        274688443,
        // @netocoelhoo
        276794985,
        // @DafneOrion
        277586067,
        // @danealencar
        278126758,
        // @CostaMarinara
        278549268,
        // @AllanPombopdt
        279092433,
        // @InspetorRobison
        280306544,
        // @mi_andrews
        284903425,
        // @victorsalatiel
        285614672,
        // @maicolmed
        286427700,
        // @gabrielaorttiz
        287219048,
        // @marcelomaranata
        287876093,
        // @anaacarlinha
        288430204,
        // @humberto130
        288512944,
        // @delegado_waldir
        288735634,
        // @laualencar_
        288761063,
        // @ZeRobertoLula
        289318056,
        // @waltercamargo40
        289521136,
        // @coelho_rodrigo
        290204659,
        // @fefrancischini
        290475286,
        // @maitebrusman
        294065810,
        // @alexdapiata
        295697900,
        // @gamanaiton
        295949375,
        // @RinaldoJunior40
        295964934,
        // @cleniltonsc
        297025171,
        // @DepSostenes
        298308683,
        // @CarlaPrataReal
        299254693,
        // @titotorresmg
        299923575,
        // @depkleberRN
        302056921,
        // @LopesCancadoAdv
        302235657,
        // @ricardopinaffi
        302725028,
        // @LelinhoLopes
        303942993,
        // @orlandosilva
        304092926,
        // @marcosjorgebv
        306253241,
        // @fabiarichter
        307557586,
        // @gleicejanems
        308780059,
        // @guto_schiavetto
        309004600,
        // @SamiraDaud
        310033093,
        // @GildeteAlves
        310502840,
        // @TercioTinoco
        311616488,
        // @paulobilynskyj1
        311720533,
        // @katiabacelar
        312252804,
        // @depandresoares
        313466411,
        // @WedersonLopes
        313684933,
        // @franciane_bayer
        317784577,
        // @_sergiosouza
        317973567,
        // @danielvalencapt
        318019551,
        // @depmaracaseiro
        321414691,
        // @NonatoSampaio
        321556855,
        // @loureirocris
        322406139,
        // @jorgeviana
        325131009,
        // @AnibelliNeto
        327327238,
        // @MartaVitorino
        327423765,
        // @OtoniDepFederal
        330770436,
        // @patriciamelobr
        331174978,
        // @MajorVitorHugo
        332324517,
        // @RequiaoFilho
        333720455,
        // @Fabinho_Gaspar
        334978230,
        // @simboramudar22
        337269106,
        // @_soedi_
        339854006,
        // @marisa_lobo
        340331807,
        // @danniellibrelon
        340748015,
        // @MariliaFreireAM
        343512877,
        // @MarceloBelinati
        345512946,
        // @SertanejoUFC
        346580194,
        // @edusantosdf
        348089005,
        // @leilafonsecapb
        349053302,
        // @paulinhoramosap
        349163261,
        // @DenianCouto
        350980280,
        // @JanderBrum
        351008004,
        // @paulapreta50
        351272601,
        // @RenatoLoffi
        351817280,
        // @Haddad_Fernando
        354095556,
        // @Biakicis
        357030742,
        // @wanderleyporto
        358996581,
        // @andrefernm
        360231929,
        // @PROFAVILETE
        360649695,
        // @milkleileite
        365210575,
        // @Thiagofreitaz
        365552309,
        // @andresalineiro
        367519089,
        // @Ana_claudiapb
        370942708,
        // @RONIESILVA15
        373465468,
        // @coroneljunior
        381045128,
        // @ieda_chaves
        389660780,
        // @RichardSelvagem
        394195842,
        // @depbosco
        396265009,
        // @JorgeFrederico2
        398245257,
        // @fabigoulart30
        399159303,
        // @FeFrancoOficial
        399589593,
        // @hugosilva63
        402050446,
        // @WMello69
        402958613,
        // @brena_dianna
        412508127,
        // @faf_freitas
        412669574,
        // @_robertoluiz
        415073795,
        // @zeliomota
        420614002,
        // @AMonicaFacio
        422814320,
        // @Carlos_Gaguim
        423706952,
        // @Edianefolle
        423874716,
        // @vitordeangelo
        424455116,
        // @BalbinottiFilho
        428247512,
        // @RealNabor
        429039681,
        // @celaocordeiro
        429940274,
        // @joicealvarengab
        433833555,
        // @instrutormarcio
        437397340,
        // @RenataAbreu2020
        441108081,
        // @wilsinhodatabu
        442693955,
        // @AllysonBezerra_
        444925483,
        // @LorruanaM
        456789752,
        // @CarlosMoises
        456795842,
        // @drpaulocruz
        459276565,
        // @soudaniellapb
        459320302,
        // @JulioCesarRib
        459681629,
        // @marcosbrazrio
        460343731,
        // @advemanuelbueno
        465613391,
        // @bellagoncalvs
        469918968,
        // @crisbrasilreal
        472033751,
        // @andreawerner_
        475996406,
        // @leopratesba
        480604517,
        // @IanBlois
        480755252,
        // @xerifedoconsum
        483269816,
        // @chicorafa_s
        487108193,
        // @pedroluislongo
        487622592,
        // @LPescinelli
        492334281,
        // @brunosouzasc
        494268633,
        // @brauliolaranovo
        505339844,
        // @caiocordeirotv
        521514005,
        // @glauberbastos_
        527053144,
        // @MoisesSantosAc
        538269241,
        // @JulianaBRedivo
        545696318,
        // @mariarosassp
        554803332,
        // @alcyvania
        556852639,
        // @limmajr168
        565889975,
        // @ThammyReal
        578495086,
        // @mariohildebrand
        580176611,
        // @OthelinoNeto
        583377940,
        // @HelioWirbiski
        589375704,
        // @adautosoutoms
        596446640,
        // @AfonsoFlorence
        599427558,
        // @CiceroSimplicio
        604095777,
        // @lucaspavanato
        608452346,
        // @Jasson_Goulart
        608730390,
        // @MarioEsteves2
        610309440,
        // @pastorhenriquev
        617468983,
        // @MateusWesp
        618506366,
        // @elmanooficial
        626602522,
        // @fabiogov55
        630758039,
        // @rosemodestoms
        632251844,
        // @helenaduailibe_
        632737286,
        // @andrewleal2
        636590052,
        // @WesleyCasaForte
        637247323,
        // @charlessantosmg
        709095821,
        // @Cidinho_Santos
        745333897,
        // @mitchellemeira
        746090918,
        // @FKrelling
        753495397,
        // @nikolas_dm
        758264276,
        // @suelenmarques06
        796882578,
        // @bneydavid
        797062418,
        // @MottaTarcisio
        799260530,
        // @JenirNeves
        813802178,
        // @PedroDeyrot
        857054846,
        // @carlaopelobem
        893975196,
        // @Isoldadantaspt
        999393290,
        // @dr_nesio
        1008361322,
        // @MarcelloPaula
        1008742886,
        // @SHEILAKLENER
        1038802238,
        // @rodrigostm10
        1038849745,
        // @gabrieldiedrich
        1059637784,
        // @xambinhoes
        1068257582,
        // @depjpassarinho
        1071798366,
        // @MacaeEvaristo
        1075036110,
        // @tatianehelena81
        1075326786,
        // @moisesselerges
        1081998169,
        // @deproosevelt
        1084884007,
        // @depjanetedesa
        1094959356,
        // @NeumannJarbas
        1124876318,
        // @McSmithOriginal
        1127018335,
        // @cacateixeira45
        1130957935,
        // @daltonlueders
        1162799238,
        // @prof_juniorgeo
        1226451780,
        // @marcoswesleymw
        1308817182,
        // @alexceoficial
        1314840228,
        // @ThaysBieberbach
        1316758495,
        // @zecarlospt
        1325494376,
        // @boettchermarcos
        1339028060,
        // @CARLOSVALADARE7
        1356677952,
        // @D_GoretePereira
        1362596354,
        // @RobertoRocha_MA
        1436541721,
        // @mickasevalho
        1461892208,
        // @reginetebispo
        1485556436,
        // @KimKataguiri
        1494658207,
        // @leticiamattossc
        1498645730,
        // @deyvidbacelar
        1526183576,
        // @bublitz_a
        1532130170,
        // @Luiza_RibeiroG
        1544201047,
        // @pedrocampospe
        1546304018,
        // @natthpaccola
        1570635661,
        // @NegrahLima
        1571332381,
        // @oficialdmarques
        1604525460,
        // @carlosfportinho
        1612753909,
        // @marciopachecopf
        1613063005,
        // @renancalheiros
        1650330319,
        // @GilvanMaximoOfc
        1651852124,
        // @carlosedrsantos
        1685646356,
        // @alcidesfernm
        1687046911,
        // @Brandaveneno
        1710385291,
        // @chaficlays
        1725112411,
        // @bombeirorafa
        1733073672,
        // @dr_furlan
        1844887189,
        // @josaqueirozpt
        1848767107,
        // @margabuzetti
        1854699607,
        // @ReginaldoVeras
        1864033711,
        // @lucianomattosmp
        1892291491,
        // @valmirdesergipe
        1960681207,
        // @Alexandresantan
        1964899736,
        // @helinhocastro
        1977089148,
        // @DepAntoniaLucia
        2161527577,
        // @JackRochaes
        2174078260,
        // @juniortunao
        2178147073,
        // @talitavazbh
        2208038935,
        // @lidiamourac
        2210601883,
        // @LUANARUIZSILVA
        2216639570,
        // @DepFederalMoses
        2217650233,
        // @andredopradosp
        2250697531,
        // @ZeniteRosa
        2289590857,
        // @BahMatteuss
        2293776768,
        // @deltanmd
        2296138146,
        // @BrunoCarianha
        2299610603,
        // @MatSchilling
        2310607019,
        // @delmartharocha
        2312487756,
        // @ProfClaudioBran
        2352495202,
        // @DavidAlmeidaAM
        2353403137,
        // @capitaoassis10
        2359974485,
        // @marciolabre
        2401917402,
        // @sheikhrodrigo
        2429206439,
        // @OmarAzizAm_
        2445921702,
        // @Francadf_
        2445938091,
        // @Alfredoficial22
        2474258532,
        // @GenPeternelli
        2491980048,
        // @yuriarrudam
        2495584641,
        // @simonetebetbr
        2508415207,
        // @prof_rosaneide
        2523520542,
        // @rosecipriano_
        2523530742,
        // @tiagossimon
        2525023980,
        // @DelegadoOlim
        2540255982,
        // @gleidept13
        2544622051,
        // @leila_0ficial
        2560238086,
        // @viagensdaiw
        2572998767,
        // @Mata4Adrianada
        2580412784,
        // @anadogasoficial
        2583179096,
        // @depheliolopes
        2585621855,
        // @profsoniameire
        2604536294,
        // @depjorgesolla
        2605560932,
        // @mvictoriabb
        2609799432,
        // @barbosinhams
        2612421128,
        // @_newtoncardoso
        2613918878,
        // @EderMauroPA
        2632802395,
        // @paulo_litro
        2647646724,
        // @LuizianneLinsCE
        2666819714,
        // @LulaOficial
        2670726740,
        // @NiltoTatto
        2674564802,
        // @ReginaFortunati
        2674769546,
        // @depjoaodanielpt
        2690172127,
        // @marcoaurelioITZ
        2691147288,
        // @cafabrini
        2691507383,
        // @UalidRabah
        2712571453,
        // @LucasVergilioGO
        2717062461,
        // @netto_expedito
        2717313965,
        // @tiaomedeiros
        2732504341,
        // @luthrebeloPA
        2741279361,
        // @crismachadotity
        2762117711,
        // @DepSanderson
        2767167039,
        // @RenanPaesSP
        2784935916,
        // @carteiroreaca
        2797010717,
        // @MarleideCunhaRN
        2820378598,
        // @mqueiroga22
        2823869072,
        // @HigaWagner
        2830971208,
        // @AlbertoMaiaf
        2834745257,
        // @RubinhoNunes
        2838953716,
        // @daianasantospoa
        2858823694,
        // @yagorVieira
        2868228117,
        // @_akalicia_
        2879108776,
        // @paulolemosap
        2881293743,
        // @manascimentogo
        2892653500,
        // @LizianyM
        2894163148,
        // @RUBENSCANTUARIO
        2903026858,
        // @andreicastroba
        2925491427,
        // @KuhlmannJean
        2927136550,
        // @NoletoMBL
        2938978041,
        // @DepJuscelino
        2970617333,
        // @neioluciofp
        2977449981,
        // @cabobonadiman
        2977624732,
        // @guipaoficial
        2979670457,
        // @depjosenelto
        2995610423,
        // @isaakalmeida93
        2997861520,
        // @carlosveraspt
        3010698441,
        // @miguelcosta46
        3015267897,
        // @jbittencourtjr
        3018188571,
        // @mendanhagustavo
        3018780587,
        // @VillaMarcovilla
        3021556996,
        // @lvaroDomingues1
        3025314889,
        // @karlosbernardoA
        3029604164,
        // @tatianeruas
        3030593279,
        // @depniltonfranco
        3044931389,
        // @AdrillesRJorge
        3060235071,
        // @jackson56786523
        3084537202,
        // @DepVitorLippi
        3092560931,
        // @rosangelawm
        3096479489,
        // @ericgustavodf
        3111776739,
        // @TeonilioBarba
        3119378914,
        // @otavio_camp
        3125324049,
        // @jarbassoaresjr
        3125532669,
        // @zuleidequeirozf
        3130842411,
        // @doutor_vicente
        3130887358,
        // @DrLeonardomt
        3131609429,
        // @senadorasoraya
        3167874665,
        // @meire_cruvinel
        3205786257,
        // @andre_fufuca
        3254270357,
        // @Marcio_Honaiser
        3294107902,
        // @deputadopriante
        3305760711,
        // @marcos_reategui
        3306282352,
        // @elikatakimoto
        3316550554,
        // @ClaudiomirCast1
        3335733075,
        // @drjeanfreire
        3342622547,
        // @castrothulio
        3354383374,
        // @PbnConcursos
        3357932231,
        // @brasleiroluc
        3366080079,
        // @moisesbrazpt
        3373574517,
        // @caformigoni
        3385089700,
        // @kleybe_morais
        3512854216,
        // @carmelonetobr
        3662771592,
        // @OPaiakan
        3674682197,
        // @delegadanadine
        3744465381,
        // @ninasouzarn
        3904595243,
        // @FelipeMichelRJ
        4026638189,
        // @geivissonvieira
        4028627062,
        // @andersonrib_rj
        4031541093,
        // @paulo_mavignier
        4043244676,
        // @depsorayasantos
        4052741685,
        // @zeaugustonalin
        4056653428,
        // @GeraldoStocco
        4057530885,
        // @Vinylobianco
        4077498658,
        // @depzegeraldopt
        4204580127,
        // @_AlessandroSE
        4250596815,
        // @IrataAbreu
        4309564575,
        // @DepLucianoMDB
        4350613047,
        // @josuelsantosbch
        4493572241,
        // @CarolDeToni
        4566967516,
        // @brunocunhablu
        4648157621,
        // @FofaBorges
        4775264669,
        // @andrebrandao_rj
        4820124735,
        // @SandroFilhoBA
        4836857632,
        // @ederborgesbr
        4871172143,
        // @emersonosasco
        4892851287,
        // @AdvogadaDoPovo
        4895047809,
        // @jumildemberg
        4899470602,
        // @joaoazevedolins
        698129524106653697,
        // @AndreaOficial55
        703604819538407424,
        // @PaulinhaQuint
        703685122265116673,
        // @juvircostella
        707214837584105472,
        // @JuninhoSinono
        711521612219158528,
        // @MARTACLERIALIMA
        713343399898820608,
        // @drlaudicerio
        713734821747560448,
        // @OctavioSampaio_
        714887920931442690,
        // @MichellePSOL
        714891302723264512,
        // @MiguelSRossetto
        714958583239151616,
        // @BentoLeiteML
        717529076974620672,
        // @SauanRockenbach
        719724675908124672,
        // @alvaroporto_pe
        719998490831634432,
        // @LiaGomesCE
        724003686863740933,
        // @ViniciusFerroC
        726243481669242880,
        // @emersonjarude
        726777982115799040,
        // @lpbragancabr
        728281672731471873,
        // @brisabracchi13
        728292397130592256,
        // @zoemartinez_05
        730865469054435328,
        // @djalmaneryneto
        732299122242359296,
        // @ErikakHilton
        738143559920934912,
        // @dreltonjr
        739270494629842946,
        // @cabo_senna
        744609688415789056,
        // @rjdouglasgomes
        745611833248186368,
        // @jarir_pereira
        746804180430487554,
        // @doorgalandrada
        748233031337545728,
        // @paulobrant_
        750386404417495040,
        // @TamirFelipe
        750744436271964165,
        // @coronelrochase
        752300463693914113,
        // @femirandapsol
        753276824743018497,
        // @samuelsalazarPE
        753589568314703872,
        // @jeffersonlimapa
        757671198125858816,
        // @JanainaDoBrasil
        759001618884939776,
        // @gauchodageral
        760848309892161536,
        // @ronaldodimasto
        761550012652322816,
        // @mazer_pg
        762337048858595328,
        // @toinhocarolino
        762649334504689664,
        // @lindabrasilse
        764288327713452032,
        // @angelaamin11
        766624608338477056,
        // @LeonardoBalbi14
        767499420384497664,
        // @kauan_poubel
        770098743412752385,
        // @vanessapstubh
        770298023582765057,
        // @fernandogorgen
        771705262671556608,
        // @pereira_lenilda
        779061594970025984,
        // @Euallephhillz
        780616664601604096,
        // @MariaConstantin
        782293493364457473,
        // @taguayork
        785545895748173826,
        // @deputadodocarmo
        793261830072311812,
        // @RafaelMarcal33
        793424665431666688,
        // @DaniloBalas
        797527121879068672,
        // @iyagomedeiros
        799344884197064704,
        // @SocorroLac
        803633234705645568,
        // @CatharinaDon
        806302110899994636,
        // @DragaAlana
        806523898917515266,
        // @Jorgepinheiroof
        807903184576532480,
        // @ruanmartinsjp
        811160203483893765,
        // @fabianodaluzsc
        813543058457423872,
        // @mariana_psol
        818583425149902849,
        // @amorimvivian_
        820867290749161472,
        // @goura_nataraj
        821001145359409152,
        // @VictorRuizSP
        823586870864932866,
        // @ProfessorEuler
        824285052590845955,
        // @luizfernandopt
        826789430425960450,
        // @angelocoronel_
        829391158208032768,
        // @AdelitaMonteiro
        830144764360192002,
        // @ranallipf
        832322309436403712,
        // @OseiasVarao
        832603702737379329,
        // @depeniotatto
        834065907743854592,
        // @peumendonca23
        839127759955841024,
        // @DaSilvaBelford
        841440085035896834,
        // @GeneralGirao
        841700087143288832,
        // @viniciusaithsp
        845046382910197760,
        // @_LeandroBello
        850113810681802756,
        // @Guischleder
        850728101868863490,
        // @WillaceSouza
        850775334362505216,
        // @vivitobiasms
        852960064159842305,
        // @Marcio_Canella
        860275250830999554,
        // @rodrigodiasbsb
        863812402680397826,
        // @elke_pimentel
        864274647583469568,
        // @MiqueiasS0ares
        868677911674597376,
        // @DanielaCarneiro
        869919798767091712,
        // @BuchiOgier
        874408561572552704,
        // @renato_battista
        877159534430650369,
        // @deputadohalley
        879906917371564033,
        // @pedroalmeidace
        882539177757298689,
        // @AndreCeciliano
        885883830489554944,
        // @eucricielle
        887083474104049664,
        // @paulosafederal
        889464426222563328,
        // @brunopedralva
        889497158491271169,
        // @jonesmanoel_PE
        889628930436730881,
        // @MayconRobertoPR
        893177406705553410,
        // @izadutrabr
        893178307591770112,
        // @CatiaColombo2
        894671613576269826,
        // @NiccSan
        898362443717627904,
        // @meuamigojoao
        899602419302240257,
        // @v_medioli
        902634025566732288,
        // @Diego_Bentim
        903169689202954241,
        // @todandara
        904842960931565568,
        // @natalimamt
        905965883952164864,
        // @AlcyPinheiroCE
        909049597091356673,
        // @coroneljonildo
        913597378002866177,
        // @deborapsol
        915203945479458818,
        // @leonidio_boucas
        917373635605786624,
        // @cabojunioamaral
        917727546615193601,
        // @NoemiaTiago
        918524953561100288,
        // @wanderley_vieir
        921969720546480128,
        // @GuajajaraSonia
        924978508224516096,
        // @valentinarrocha
        927752925031620608,
        // @MajorMecca
        931022287066804224,
        // @TorminCassiana
        931087564064329728,
        // @ptalfredinho
        931140401293070337,
        // @julianopsol
        937683426882326529,
        // @rigoni_felipe
        939423395376136192,
        // @sergiokruke
        940210949943910401,
        // @fabiofreitaspa
        940623667322609664,
        // @pedrocheoficial
        947255546062745600,
        // @jusmarioficial
        948577496177561600,
        // @DpRicardoArruda
        953055428124045313,
        // @Carlos_cabral81
        956932819086913536,
        // @WilliamSiriRJ
        958357145761845248,
        // @delegadopiquet
        958833209961254912,
        // @DepJorgeEverton
        959184931552464896,
        // @ClaudioLCaivano
        962035539133177858,
        // @Fbgg40
        963928182368980993,
        // @RafaelLustoza_
        965096425313972229,
        // @XandePessoaPE
        966833467391725569,
        // @Pastorellux
        968686315326799872,
        // @rsallesmma
        971822131024748545,
        // @RNBolsonaro22
        973888139637993472,
        // @dep_paulinha
        973982865997385728,
        // @ORicardoDaKarol
        976888734267445248,
        // @Diegoferdfc
        977195483507707904,
        // @MoRosenbergSP
        977579494344142848,
        // @AndreTrindadeRP
        978076973300879360,
        // @ZeCocaOficial
        978602905690427392,
        // @jairfarias_to
        979750671728685056,
        // @julialucydf
        980846559444291585,
        // @Indianarae1
        981429809413873664,
        // @Fausto_Pinato
        981500250602041344,
        // @oficialigortimo
        981604712238780432,
        // @ChristianProf_
        981899518244544514,
        // @Renatoafjr
        982217430297554945,
        // @fernandomaximoX
        983360917227364353,
        // @RenilceOficial
        984221723544444928,
        // @eng_angelo44
        984522534623301632,
        // @israelsantosap
        985962825909723141,
        // @Dep_GilPereira
        986284866533842944,
        // @IndiaraNOVO
        986396254287646721,
        // @leaolais_
        986659343436255232,
        // @ptmatiaspedro
        986903953014128641,
        // @leiladovolei
        986978033159626754,
        // @DougjButzke
        988809130256273408,
        // @paulambelmonte
        989854043370606592,
        // @CaboDaciolo
        989899804200325121,
        // @GutoZacariasMBL
        991090809578708992,
        // @RJRogerioAmorim
        992223431906287616,
        // @KerexuOficial
        992801222926196736,
        // @danimontpsol
        993146341294596096,
        // @kekabagno
        993288307625943040,
        // @andreiadejesuus
        994268335486504963,
        // @Alessandrojesu_
        996849375635820544,
        // @prdinhosouza
        999747749364076545,
        // @thiagoavilabr
        1000525959257378816,
        // @Izalourenca
        1000724258740473856,
        // @tabataamaralsp
        1001251931812220928,
        // @NatanSperafico
        1001476076466593792,
        // @zuccors
        1002182052341534720,
        // @romulobuldrini
        1002646368496865282,
        // @paparicobacchi
        1003614554394415104,
        // @jmtavaresz
        1004078050521305088,
        // @DudaSalabert
        1004511711251099653,
        // @FreireJocimar
        1005181143627522049,
        // @ferreirinharj
        1006202251990437888,
        // @ZeDirceu_
        1007663803948027904,
        // @annakarinapsol
        1009608950176796677,
        // @CapitaoContar
        1011667518728089602,
        // @f_francischini
        1012066500373557253,
        // @EricaGorga
        1012086799328522246,
        // @gabino
        1012469182330355713,
        // @Trom_Petista
        1013494732834639874,
        // @danicunhario
        1013945423676010497,
        // @limmapiaui
        1014568888078651392,
        // @costa_major
        1015188576584298497,
        // @acaroldartora
        1016697971843502080,
        // @CarlosBurigo
        1019197414136320000,
        // @danielibalbi
        1020645096226795520,
        // @RomeuZema
        1020798087449776128,
        // @EduGiraoOficial
        1024403315164160000,
        // @Laurez_Moreira
        1025352673292378113,
        // @juliadecastrobr
        1025519861592666113,
        // @zenaidern
        1025870063461720064,
        // @deppimentel
        1026636587298443265,
        // @EuclydesPetter
        1027187280438546433,
        // @Pataielloo
        1027646933110804480,
        // @MarcosFelipiPoa
        1028075839269888001,
        // @MaisaMitidieri
        1030455388678889472,
        // @professorjoziel
        1030551918127529990,
        // @KenjiNohama
        1030577196598001664,
        // @BocaAbertaOf
        1031914738933018624,
        // @leomartins45rs
        1032017342937661441,
        // @vereadornetuno
        1032654017418211330,
        // @SalvinoOliveir1
        1034050960262418433,
        // @victorhugoforte
        1034054725107363840,
        // @RafagninLuciana
        1034079788162535424,
        // @erikaamorimce
        1034138653147242496,
        // @AuricchioThiago
        1034858280231809024,
        // @victoriogallimt
        1036259095458795526,
        // @AmauriRibeiroGO
        1036681658760658945,
        // @NeymarPesadao
        1037281805118910466,
        // @FaleiroAirton
        1037384667836637185,
        // @phbarroso45
        1039361882845528064,
        // @AlanaPassosRJ
        1039881241301016576,
        // @DouglasGarcia
        1040704983358955526,
        // @RenanSantosMBL
        1042601099566436352,
        // @Daysehansa1
        1044658383922638849,
        // @CavalarEmanuel
        1047079027696177152,
        // @gutopfonseca
        1048016722433900550,
        // @tome_abduch
        1049634819028717569,
        // @raphaelbarrabr
        1049870508643233793,
        // @ContaratoSenado
        1050121324436307970,
        // @EuJorgeMiranda
        1051329992662142977,
        // @Cristian0Engel
        1051904895106908161,
        // @manumirella_
        1052932498181840898,
        // @depmgualberto
        1053123917785808901,
        // @RafaelDemarchi5
        1053334763858214912,
        // @deputadatalita
        1055640991200411648,
        // @cleitinhotmj
        1057231251743170562,
        // @gilson__marques
        1057602315664924673,
        // @delegadasheila
        1058010509256126464,
        // @mauricio_lindol
        1058341993372360705,
        // @evandro_stacruz
        1059551999258255361,
        // @veronicalima_ve
        1059554967600685058,
        // @pinheirinhomg
        1060134845043666945,
        // @RuthVenceremos
        1061207070412886016,
        // @giordanmes
        1062407588724252673,
        // @pluviapt
        1062505159824150530,
        // @capitaocarpe
        1062678892216020992,
        // @FlaviaHellen_13
        1062871543007662080,
        // @rafaelsimoesmg
        1064322572261699585,
        // @clarissatercio
        1065379080084865024,
        // @professorabebel
        1065745997391949824,
        // @GIBERTOPINHEIRO
        1066799808172695552,
        // @evandroaraujodf
        1069259896892329984,
        // @DelHelioBressan
        1070740683386957825,
        // @wilkerleaods
        1073662759009771520,
        // @marxbeltrao
        1074651587140902913,
        // @capalbertoneto
        1076135802969772032,
        // @tarcisiogdf
        1078618844007157761,
        // @EduardoBraide
        1080814654195204096,
        // @MarcosA_Sampaio
        1080911369447399430,
        // @capitao_alden
        1081245039920144384,
        // @deputadomoraes
        1081341581314150400,
        // @DerriteSP
        1081517956964802561,
        // @proanalucia
        1081700266410426369,
        // @ingrapsol
        1081957744725381120,
        // @matheuspggomes
        1083017623837777921,
        // @Khalill_gui
        1083098735675084800,
        // @PEDROCO13904182
        1083745378489589760,
        // @DamaresAlves
        1083774484123975680,
        // @BennyBriolly
        1084272914202091520,
        // @majorfabianadep
        1084712593292443648,
        // @waltinhofoguete
        1084870539666247682,
        // @wilkerbarretoam
        1085537170276958208,
        // @ArthurLira_
        1086390169970884613,
        // @IsaacRicalde1
        1086404913314385920,
        // @emanuelzinhomt
        1087326506559389696,
        // @DaviBenevides
        1090741557383311362,
        // @issam_saado
        1090980245098979328,
        // @DelegadoFurtado
        1091410188110831616,
        // @PlinioValerio45
        1091691201844207617,
        // @joaorenatobr
        1092217577026318337,
        // @doriel_barros
        1092446595889745921,
        // @joaoluizam
        1092549114762539009,
        // @karensantospoa
        1092781921560641543,
        // @betopereirams
        1092816222012534784,
        // @biologiagabriel
        1092879030465032192,
        // @doutordanielpa
        1093086549825208321,
        // @ZequinhaMarinho
        1093223195258298368,
        // @enfermeiranaza
        1093488527583690753,
        // @igortavaresmg
        1093510541404971008,
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
        // @DiogoTalento
        1096490764677320704,
        // @ReMesquita1977
        1097042489813409792,
        // @DFDanielFreitas
        1097498693719199744,
        // @dilvandafaroPT
        1097822283387809793,
        // @joaocardosobr
        1097888011650502656,
        // @thalesdacostaal
        1098031305818812416,
        // @drmarcelmuscat
        1098089617209913344,
        // @CelAlfredo
        1098328459825287187,
        // @helio_ver
        1098911814824411137,
        // @DelegadoFreitas
        1098981858321276929,
        // @benesleocadiorn
        1099003656064643073,
        // @BlogdoSavio
        1102579349939724288,
        // @FBarcellosSC
        1103370630827855872,
        // @AcilonGoncalves
        1103616211311570944,
        // @depPedroLucasF
        1104120809482866688,
        // @jurailtonsantos
        1104747705434288128,
        // @PollonMarcos
        1105081277135417345,
        // @GabrielDosAnima
        1105170906987593728,
        // @CoelhoAntonioPE
        1106161642486796288,
        // @rodrigomoraisof
        1106200795098292229,
        // @SargentoBetania
        1107097837194694663,
        // @robsonjs_ofc
        1107238951990059008,
        // @DelegadoZucco
        1107381354680012806,
        // @drfranciscopi
        1108052690783928322,
        // @DelegadoCaveira
        1109546175982698496,
        // @leolupi_rio
        1109869057153683462,
        // @DrCesarMello
        1110177096373100545,
        // @ToniettoChris
        1110626741314306049,
        // @keilapereirasp
        1111421096501432320,
        // @JoaoCampos
        1112804630453501953,
        // @SF_Moro
        1113094855281008641,
        // @deputadothiago
        1113811245378015232,
        // @keitlimasp
        1114595200113029120,
        // @CaldeiraJos1
        1114734016811474946,
        // @DeputadoRatinho
        1115425047135571974,
        // @faustojr_am
        1116362076459601920,
        // @DeputadoLemos
        1116429709049573376,
        // @10ronaldomartin
        1118164353121968129,
        // @guitodeschini
        1118455651901083648,
        // @deputadaniella
        1118481835376431104,
        // @flavionpi
        1118489062866849793,
        // @NajaraCosta_
        1118899291983175680,
        // @italomoreirasp
        1119731927991443462,
        // @NelPiaui
        1120084261917351936,
        // @henriquecesarr
        1121117485900730368,
        // @guicolombosc
        1123714030391132161,
        // @emanuelvenzo17
        1124016358893785089,
        // @AltairMoraessp
        1124045706883473408,
        // @eunatashapoa
        1124140833547141121,
        // @GilbertoCattan1
        1125062220302385153,
        // @Jeffers67876956
        1125132737491480576,
        // @RobertoJustusPR
        1125404580374822912,
        // @FernandaLoubac2
        1125416190199967744,
        // @doutorfran
        1126519869053251584,
        // @eduardo_borgo
        1126708527274315776,
        // @diretoreliasjr
        1130259444326109184,
        // @fabiosilvadep
        1130509226281975814,
        // @GersonSPires
        1130572020901699585,
        // @gilmarpetrolina
        1131947049690255360,
        // @brunolessarj
        1131984902390460416,
        // @rosamst_
        1132357062703439873,
        // @DeputadoPoubel
        1135964153309478912,
        // @Fabianohorta_
        1137809351387865088,
        // @vinicios_betiol
        1137885440768450561,
        // @vilmareisfem
        1137930985109110785,
        // @PapoDoCappa
        1140263037049475072,
        // @FernandaSixel
        1140840515124047873,
        // @DirleteP
        1141306501019119616,
        // @georgebastos30
        1141394024860966912,
        // @delegadotayah
        1141737559816585217,
        // @CarboniRogerio
        1143607219302387712,
        // @lesinhaduarte
        1144077937228034048,
        // @felipebecari
        1145863684511674369,
        // @luisedulelis
        1148639803908472832,
        // @pedrorochafilho
        1151964529896644620,
        // @ShirleyCruz22
        1152873734157393922,
        // @washingtonban
        1154733508088270849,
        // @DacunhaDelegado
        1155962982486171648,
        // @AlanLopesRio
        1157984524116156416,
        // @torinomarques
        1158806983925030913,
        // @BertolucciGab
        1159503401463484417,
        // @karlasarney_
        1160505413697253376,
        // @erickdenil65
        1161802414921527296,
        // @JalserRenier
        1164616428944801793,
        // @jofariasm_
        1167172974451073024,
        // @RobertoCidadeAm
        1168549394830020608,
        // @umJovemPaulo
        1169327756544479232,
        // @monicacunhario
        1171205026724929537,
        // @jadearomero
        1171411853282557952,
        // @prjuniortercio
        1171959927771975680,
        // @alice_psol
        1174767854459195393,
        // @uaivittor
        1174847040406376449,
        // @carlosnovofsa
        1176225764200591361,
        // @joaobettega_
        1177185451536465920,
        // @marcimeirelles
        1177246955942154241,
        // @GalbaSheyla
        1179012777366736899,
        // @delegadopalumbo
        1179437585275465729,
        // @by_dzn
        1184622911174389762,
        // @rickazzevedo
        1184969015023800320,
        // @PauloSussumu
        1188038521501749248,
        // @FranciscodoPT13
        1188191474636206082,
        // @samueljesusjb
        1188281892229074951,
        // @atenabr051
        1188827192999972865,
        // @BrunoZambelli3
        1191788412833075201,
        // @rebecaromerobr
        1192983475663638529,
        // @ProtetorAle
        1193938478456868865,
        // @thiagomedinamd
        1196894264334127107,
        // @leninhamoc13
        1197135895469711362,
        // @JulianaBenicio_
        1197281720694956032,
        // @PettersNeto
        1198325133368340480,
        // @DeputadoCarlosH
        1199293398194233344,
        // @mariliacamposmg
        1199714008217063424,
        // @SandraLimadeVa1
        1199740816018853890,
        // @tukuma_pataxo
        1200840202194952194,
        // @Ronygabriel_ofc
        1208038091962888194,
        // @LeoSuricate
        1208544960032727040,
        // @ManuVieiraSC
        1210296676520513536,
        // @CruzOrleans
        1211689081861623808,
        // @michelbeckerbr
        1211812977579421699,
        // @mello_bandeira
        1213217041739419648,
        // @mfriasoficial
        1213876635331465216,
        // @OficialNenemAl
        1216746154626625536,
        // @JairSoutoAM
        1218904655591243777,
        // @JohnRobertPA
        1219003418347483136,
        // @cortezpsol
        1219445803854548994,
        // @mariamarighella
        1220387710462021633,
        // @BernardinoNOVO
        1223018829393145856,
        // @GuiBianco65
        1223960824403939328,
        // @VicMello16
        1224035025882140673,
        // @Vilsondafetaemg
        1224332353927073792,
        // @DrVictorAmoras
        1224505061558161408,
        // @luanlennonbr
        1224592244465905666,
        // @KlesleyGarcia
        1224615230195609601,
        // @PortaldoJose
        1224767212101283840,
        // @joaobmaresguia
        1225055364124762112,
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
        // @marinahelenabr
        1227202002066837504,
        // @VaniceMatos
        1227388838592614400,
        // @JulioCLamim
        1227572253522526208,
        // @Alfredogaspar_
        1228008951813492739,
        // @izadoradias29
        1231267069762691072,
        // @BabaTupinamba
        1232082766071767045,
        // @felipegomespsol
        1232252941681221632,
        // @amandasalecosta
        1232788071218843648,
        // @queciareismbl
        1233041635480678401,
        // @carolnunesof
        1233941640441663488,
        // @bolsonaronegona
        1234168067191648259,
        // @MassamiMiki
        1234278834519920642,
        // @juizcubas
        1234488841479884801,
        // @JacqueMoraes_es
        1235371041712726022,
        // @AndreKubitschek
        1238100206488629248,
        // @julinhodeputado
        1238202050745446401,
        // @kikoceleguim
        1238511273752563712,
        // @NaldyBianca
        1238835947661377536,
        // @eusouamom
        1238868377675980803,
        // @eduardomenegol_
        1240359378081001474,
        // @juliana_macieel
        1240694913420926976,
        // @fabriciochaves_
        1241145962082467851,
        // @barcllay
        1241746085820932098,
        // @fabiolopes_38
        1241821464111853570,
        // @cop_santana
        1241842077979283466,
        // @elianexunakalo
        1241878280841674752,
        // @EiflerPam
        1241960263886286854,
        // @depvermelho
        1242093483151839238,
        // @claudiaguerramg
        1242834650155941888,
        // @gabrielcarvalce
        1243211951863463941,
        // @samaramartinsup
        1244783574676627460,
        // @docporto
        1244954874049114113,
        // @allinyserrao
        1245187621007089666,
        // @vitimporto
        1246125973755609088,
        // @wmf_oficial
        1246641106391052288,
        // @GallinatiRaquel
        1246945909881044994,
        // @RodrigoLivra
        1247224255311499264,
        // @drluizovando
        1247287375803355136,
        // @robertlemosss
        1247578937510723585,
        // @enricolopescba
        1247916183392837633,
        // @leodacostas
        1248042609731387392,
        // @victorjansendf
        1248682133943717889,
        // @a_jessicao
        1248830800281378817,
        // @CoronelFernand9
        1249769581725573121,
        // @lucaspoleseES
        1250037140647481344,
        // @tuannepsol
        1250165291558068230,
        // @VandaMonteiro_
        1251221027822211073,
        // @jaziel_dr
        1251558672821600259,
        // @cirineu_costa
        1251676296842805257,
        // @Victorinogustav
        1252237633008238593,
        // @fcordeirosc
        1252561439686037506,
        // @ProfDrMauroRosa
        1252709404975169538,
        // @fabio_schiochet
        1253698500401008641,
        // @ivoneidecaetano
        1253699087209127936,
        // @tyago_hoffmann
        1254019417148661760,
        // @peritatirotti
        1254073103661088768,
        // @tanielmacedo
        1254084527422689282,
        // @deputadaleandre
        1254103540387254274,
        // @israelcosta_rs
        1255134132889214984,
        // @MoraisDino
        1255145793507348482,
        // @beckhausersc
        1257020913893195779,
        // @DraSilvana2
        1257476064219140097,
        // @GiovaniMattoss
        1258511929146040320,
        // @najuliaribeiro
        1259624231765184514,
        // @adailfilhoam
        1260696151029940225,
        // @camilajarams
        1261423487585005575,
        // @brunoseccobr
        1261452346028105728,
        // @ze_haroldo
        1262408615564099596,
        // @luannasantos_13
        1262645801936879617,
        // @nise_dra
        1262944997118181385,
        // @mikaelgringoPE
        1265422039260762113,
        // @hana_ghassan
        1266449427985829888,
        // @professorarita_
        1266477339413749761,
        // @mariadoscamelos
        1267127677468753920,
        // @Junacamara
        1267549843586768896,
        // @aanaelisast
        1267822695195930630,
        // @ChirleyPankara
        1268292556967874566,
        // @maiarafelicioo
        1269374979222765568,
        // @manupeloES
        1269632201760690176,
        // @DRodrigueiro
        1269706255347650565,
        // @ThiagoResiste
        1270774020808626176,
        // @juliermesenav
        1271216272148176896,
        // @GilvanDaFederal
        1273299541786349569,
        // @carladicksonrn
        1273322623380983815,
        // @MucioBotelho
        1275549634757365760,
        // @LucasCaregnato
        1275636866075803648,
        // @GiorgiaPratesMP
        1278774539942584322,
        // @CrBeraldo
        1278784862313472005,
        // @victorcarvamt
        1280120303818072064,
        // @JuniorGeraldo_
        1280224335307907077,
        // @ChinaoRuiz
        1280315144338313217,
        // @FeCurti13
        1280517974449893377,
        // @wagnertavaresrj
        1281590900834078720,
        // @jonasreispt
        1281598785081180162,
        // @KodamaThiago
        1281980054730407936,
        // @daniel_sucupira
        1283586848019951621,
        // @adrianaraujomg
        1284456064432377856,
        // @GlauberPoubel
        1284849642895740930,
        // @DepChrisostomo
        1285226176953360384,
        // @eugenialima_pe
        1285282863471001601,
        // @daniportelape
        1285295088525082631,
        // @Sonaira_sp
        1285299871340167173,
        // @euLiviaNoronha
        1285729712795525125,
        // @ladisouzams
        1286438113896796170,
        // @ericodonovo
        1287396715331452932,
        // @sorriso_elisa
        1287490158510723072,
        // @prjuniortrovao
        1288544254856507392,
        // @majorvitorsa
        1290713462285500417,
        // @MatheusLaiola
        1292497353166004227,
        // @DepCoronel
        1295725787790942208,
        // @rzampieri22
        1295750020554203136,
        // @joaoccoser
        1295830662721736711,
        // @KuertenRoberto
        1297864783841103877,
        // @gustavosefer
        1298369792169127939,
        // @junynhomartinss
        1298413776715358209,
        // @dinhodowsley
        1298603065457680385,
        // @luladafonte
        1300279136620085249,
        // @telma_rodolpho
        1302310926482317313,
        // @marcelinhoguima
        1303443223663325185,
        // @DimasCostaRS
        1304120510821945346,
        // @chaves_hildon
        1304134002547331072,
        // @pepeliberdadefm
        1304145702122147840,
        // @marioleonypsol
        1304884724603772931,
        // @draraissasoares
        1304911931036307456,
        // @chris4patinhas
        1305541363644104706,
        // @deppedrokemp
        1305583653934727175,
        // @damiresrinarlly
        1307532712794873856,
        // @GuguSeba
        1310253107310460933,
        // @CoronelRomualdo
        1310744586139119618,
        // @EduardoRiedel_
        1310977778574032900,
        // @PabloSilvaLira
        1311620519935045632,
        // @OdairTramontin
        1311753908549750784,
        // @gedalvaumbauba
        1312771507601498112,
        // @cassymonteiro
        1312841615111843848,
        // @delegadkatarina
        1314575606487670784,
        // @andersonlimaadm
        1318194625400737792,
        // @delegadalia
        1318672895573479424,
        // @ElianaBayer_
        1319073837485576197,
        // @anaportelams
        1319825203313184768,
        // @DeoliAnderson
        1322213572953493505,
        // @PamelaGiedre
        1325859971905572869,
        // @bfeministapsol
        1329213002051149824,
        // @paulomelo_sa
        1330155785951866881,
        // @cabomeireles
        1330393555290951681,
        // @patisborges
        1330690808090124289,
        // @_PastorDiego
        1334311453592064000,
        // @bocalomoficial
        1335685798746845190,
        // @RoseanaSarneyM
        1335942718040793091,
        // @matheuscampinas
        1338827932849090563,
        // @RenatoG05313289
        1339709453738926081,
        // @Danny57037067
        1341344192874864640,
        // @MorandoOrlando
        1341392073799380992,
        // @RenatoTaroco
        1343385618559197188,
        // @RicattoEduardo
        1344480762180022272,
        // @BrandelJunior
        1345054306614042626,
        // @ProfessoraNadir
        1345965375582851072,
        // @DepDianaBelo
        1346196281832693768,
        // @elmovazoficial
        1346929915040571399,
        // @CrisWainer
        1348639812497125377,
        // @FlavioNJunior
        1349056893235458051,
        // @adeildoreisofc
        1349210221604962306,
        // @DrGeorgeLins
        1349425371146547204,
        // @annacarolinaadv
        1349690681464385540,
        // @GuiUchoaJr
        1350924815759269888,
        // @rosanadasaude_
        1351505730323558400,
        // @Paulatitan1
        1351686391265193987,
        // @silenoguedes
        1352236209800671232,
        // @dimasfabianomg
        1355216660068761606,
        // @AnaPimentelmg
        1356434859934298115,
        // @faustinorn01
        1356762322489004032,
        // @AndrePiresDF
        1358426024250343427,
        // @RafaelFonteles_
        1358739792163389440,
        // @deputadodrPaulo
        1358880298407243778,
        // @miroteixeira
        1359157620897153024,
        // @FadelMoacyr
        1359162686228078593,
        // @robertoclaudio
        1359194831415828488,
        // @Nilvo17
        1362188334924242949,
        // @del_guilhermed
        1362442787988381696,
        // @boulosnat
        1366796662614749187,
        // @NeidinhaSurui
        1368009955912130561,
        // @_mapeoficial
        1372519763818217477,
        // @NilmaSanches
        1373064659817926661,
        // @rodrigoestacho
        1374044251475091457,
        // @PiauienseO
        1374436642375659530,
        // @VitorioJrMG
        1374502893533884421,
        // @FelipeAlecrimPE
        1375498458132537348,
        // @rodolfoms
        1376603305879736320,
        // @CarolineKalil4
        1376746550593003525,
        // @celiaxakriaba
        1379025327314329608,
        // @DrLeviMelo1
        1379424037068234752,
        // @fabiosilveirarn
        1379583456812924935,
        // @marleipr
        1380366998488645636,
        // @depprofcleiton
        1382460589625204742,
        // @apropriajulia
        1382550118314999812,
        // @PrefeitoCesar
        1383362690584768514,
        // @santanna_cs
        1385208512955957250,
        // @SylvioMenicucci
        1385213373332213778,
        // @PCO29antonio
        1387843668053200897,
        // @depmarcionunes
        1390475281316646917,
        // @davibrandaobac1
        1392451077656698881,
        // @juarezcostamt
        1394317065750724608,
        // @lucasbovesp
        1394364586346885120,
        // @MassaelB
        1395784254269841408,
        // @DelegadoMarcus
        1396055754793230341,
        // @renataseneofc
        1399347605776314368,
        // @jusoaresft
        1400452986178981891,
        // @eduacostario
        1400996132982038532,
        // @antidio_lunelli
        1402602168838938625,
        // @gu_camillo
        1405334928926162946,
        // @Sandromarttinss
        1405740717595529225,
        // @rubensuchoa_
        1406043032437264386,
        // @lelecopimentel_
        1408810159376420865,
        // @ananiasnauar
        1409194659872653317,
        // @sauloportolivei
        1409385758838906882,
        // @jrfreitasofc_
        1409935535443951616,
        // @juniorferrari55
        1409990621536919555,
        // @rafaelsaraivasp
        1410633602354794508,
        // @ProfeBonatto
        1411854540182327299,
        // @yasminvsh
        1412747070511984641,
        // @deputadodrhugo
        1413128451154866185,
        // @cozzolino_RJ
        1417893820876967943,
        // @dudasanchesba
        1422326852539015168,
        // @rjandremonteiro
        1423628533696540673,
        // @depclaudiac
        1423660842239791115,
        // @coronelbonates
        1427816945735319554,
        // @CapitaoMartim
        1430162160320188433,
        // @MauricioNeves_
        1430622957102145543,
        // @amandagentiI
        1432351330379644929,
        // @RicardoAbrao_RJ
        1433065898064191490,
        // @robertadahorta
        1434501105241792512,
        // @NFgoes
        1435915606541410305,
        // @0xcarlosh
        1437029285387243526,
        // @PeKelmon
        1437437148769226757,
        // @ScalcoDarlan
        1437758990172184577,
        // @nisia_trindade
        1438501569301991424,
        // @Jadielmoraes20
        1439280377936367621,
        // @OperadoraManu
        1439763846772703232,
        // @edilenxavier
        1441394605338021888,
        // @ivanirdosantos
        1442471110927290379,
        // @mariiluse
        1445155253880594432,
        // @renatmirandarj
        1446451214292553733,
        // @DayaneCruzPE
        1446618743350730754,
        // @DanielaadvAP
        1446856314215337989,
        // @joelrodriguespi
        1447921024117493761,
        // @atilaliraof
        1448414094441291777,
        // @LuizEduardo_RN
        1452666475844706306,
        // @PretoniDacio
        1452992283616366607,
        // @victordiaspa
        1453677155485896708,
        // @RenamTassio
        1454041097395744776,
        // @dep_dayany
        1456344779625799688,
        // @TremeaMarcio
        1456999668961906690,
        // @ChicoVieiramg
        1457348048334594049,
        // @NilceBregalda
        1457433413028356098,
        // @fala_mafia
        1458089891795976202,
        // @sargentoportuga
        1458495272716222465,
        // @josecam01577970
        1459681826700673030,
        // @schumarker7
        1461714774996230145,
        // @RicardoArrruda
        1462770465068437504,
        // @BrunoBrazKart
        1463552472052502528,
        // @KAUMAGN0
        1463557962589429767,
        // @Alexandregonrs
        1463574028468330497,
        // @padovanidep
        1464329410115514369,
        // @michelkourimbl
        1464589158035468306,
        // @marinadomst
        1465395158103597065,
        // @amarianalescano
        1468292579775348737,
        // @FernandoManso20
        1468757430632865793,
        // @Maubmarcon
        1469007240279597067,
        // @orenatomachado
        1471286780737503232,
        // @joaquimroriznet
        1472329069027119107,
        // @LuizinhoMinas
        1477024586764009478,
        // @yurydoparedao
        1478380311532736512,
        // @Anderson_ma123
        1478686290270896128,
        // @CapitaBrasil22
        1478840583481446400,
        // @diegocastroba
        1479821013009551371,
        // @firmo_oficial
        1481351815669112840,
        // @Johnatanmaravi2
        1481369125159067656,
        // @erickmonteiropa
        1481981850159652865,
        // @foliveirapr
        1483509817620709381,
        // @laisjordy
        1485332978976886786,
        // @HungaroIgor
        1487071314410188807,
        // @alexsousaam
        1488595085361041408,
        // @lucascaculajp
        1488612280241696775,
        // @deparimateia
        1488691561185619969,
        // @depjaqueline
        1488866652037029889,
        // @GayerGus
        1489108473027739654,
        // @valdiroliveira_
        1489559619118805001,
        // @AdaoPrettoFilho
        1489648399334969349,
        // @DuCazellato
        1489744612558356481,
        // @JooDePaulaDosS2
        1490037307176636420,
        // @AndersGimenes
        1491165950422495244,
        // @eribertomfilho
        1491176697445683205,
        // @SdMadalhano
        1491215424675004421,
        // @RosaRezendeGO
        1491387105913810944,
        // @joao_herzer
        1492198027620143113,
        // @neyamorimac
        1492921263299379207,
        // @cleoniceback1
        1493565805304455174,
        // @depchicao
        1496537874237468679,
        // @Delboni_Isis
        1496817240049606661,
        // @deposcargutz
        1497201650196430861,
        // @profangelapsol
        1497605679229644803,
        // @fsdecastilho
        1499529892039434251,
        // @EnfBrunoFarias
        1500910030593445888,
        // @annasebbaj
        1501892430949453827,
        // @sgtgoncalves22
        1504105286876991489,
        // @EdianeMariaMTST
        1504228964319080449,
        // @edsonferrazsc
        1505881785762304002,
        // @AlvaroJeronymo
        1506029258967240715,
        // @SKaripuna
        1506313161921777676,
        // @rhdeverdade
        1508246534370082821,
        // @bm_oliveira14
        1508783941037268993,
        // @Thais_ProfeChef
        1509245802815922177,
        // @MaalouliMari
        1509589806325579784,
        // @orleansbrandao_
        1510790014111784967,
        // @diegoandrademg
        1510954548348784645,
        // @rodrigomarcial_
        1511399052675633156,
        // @WebaNatassia
        1512098373570113538,
        // @DelRodrigoSa
        1512105284558282765,
        // @MarciaHuculak
        1512756653199872006,
        // @reillerlopes
        1513329845551390720,
        // @franze_carneiro
        1513903701483790343,
        // @mariaarraespe
        1515390927958908928,
        // @SocorroWaquiim
        1516019212145287168,
        // @PenhaBernardes
        1516033327941181447,
        // @DanielleDVale
        1516121393028644869,
        // @thiagorodripa
        1516560691410481156,
        // @andrebuenoofc
        1517514873105797120,
        // @ivanilsonrn
        1517576201338040321,
        // @leandrobasson
        1518652449371922433,
        // @drpaulomedinamg
        1518728271176871936,
        // @EDILSONLIMA70
        1518759485694779392,
        // @EdRaposo_
        1518965841261441024,
        // @luizgastaoce
        1519011181574434816,
        // @PazuelloGeneral
        1519440468949573632,
        // @drviniciuspi
        1519744542173536259,
        // @Ricardinhofoz
        1520051894319788033,
        // @Julianoomarti14
        1520087295822598146,
        // @OtacilioDeSous3
        1520962489931939840,
        // @LuizZacarias22
        1521138334256480259,
        // @Alessandrogb74
        1521179786915233795,
        // @pricarrijo28
        1521558274230919171,
        // @clementecampo15
        1522266010837041153,
        // @AnaPaulaGoffi
        1522529565733707777,
        // @coronelamadeu
        1523726346232414208,
        // @SERGIOD84944206
        1525599750854266880,
        // @SindoleyMorais
        1525834800233336832,
        // @MariaClaraMarra
        1526909238622175232,
        // @misspretapt
        1526912162194653184,
        // @IrineuCruz4
        1527267836250537984,
        // @lucianovieirarj
        1527317724979810304,
        // @ArielBrandao7
        1527695566184009731,
        // @Leonegreiross
        1528462133477916678,
        // @RicardoEndrigo5
        1529074828933746688,
        // @Marcelo00945011
        1529506446009827329,
        // @CriaCranio
        1530904551582220288,
        // @JadyelAlencarr
        1531271393219837956,
        // @Jaimenunes_ap
        1531655976008531968,
        // @ReginaluciaSi15
        1531933026208436231,
        // @eudonaneuma
        1533265972659990528,
        // @melcafariaspb
        1534926600743051266,
        // @helenadaasatur
        1536471955209076736,
        // @obrenocop
        1536700734892298241,
        // @FaoDoBolsonaro
        1537386416963039233,
        // @eliassantiagopt
        1538940264173211648,
        // @rejanepstu
        1539038049937547269,
        // @AnneMarques_AP
        1539992525980815364,
        // @eudociasenadora
        1542212008082382855,
        // @marussago
        1542233934435696646,
        // @a33lucas
        1542497617363492864,
        // @AAdautooficial
        1542590670359207941,
        // @soududaramos
        1542878687120482306,
        // @victorlinhalis
        1543943524189720576,
        // @rsodrec
        1544490645636751360,
        // @LuisaCela87
        1545437399446077440,
        // @AnibalLins
        1545770498486898689,
        // @leandrosoaressp
        1546610777519521794,
        // @GuaracyJunior4
        1546615155009667072,
        // @iossef_hassan
        1547001236163100672,
        // @AdaXavier9
        1547257445243830275,
        // @DrHenriquePaes
        1547672436933419017,
        // @drjosericardo92
        1548433495097151489,
        // @abraogodois
        1548671250679173126,
        // @ThiagoManzoniDF
        1549429329494478848,
        // @williambarros30
        1550186795333296128,
        // @julio_kuller
        1551652560951562240,
        // @DepAndreAbdon
        1552328294665781249,
        // @Gi_MonteiroRJ
        1553042102908502021,
        // @MissiasDias
        1553167499901943808,
        // @draalehaber
        1554462338715197446,
        // @pedroaiharamg
        1555552838008414210,
        // @catanhopt
        1555559206635315201,
        // @PQueirozadv
        1555860014383988737,
        // @LaironCarlos
        1556676552674394114,
        // @KleberRosa50
        1556777705387032577,
        // @josyjacob_
        1556827882516893700,
        // @VitormoreiraCG
        1557086339740405761,
        // @ScaranteRenato
        1557393886431027202,
        // @Raphael89700864
        1558100800664178688,
        // @ArianeRSAssis
        1559215037746741251,
        // @samuelvianamg
        1559598025420464134,
        // @Clecio_Luis
        1560635958994812928,
        // @BartonCutler
        1561212563836264449,
        // @StellaGaio1224
        1561545109715517440,
        // @carlosmoraestv
        1563342678011842560,
        // @vinivenades
        1564049349000183809,
        // @DeboraMenezes22
        1564101654496133120,
        // @daiane_daiane18
        1564304740850221058,
        // @jaquepetrovik_
        1565401487836061698,
        // @NMousquer
        1569113308078170115,
        // @pitmagrin
        1569391051160313858,
        // @oAllaxSiqueira
        1569761874471972865,
        // @regisethur
        1574862745325146112,
        // @weibetapeba
        1575549757900132379,
        // @14luizfranca
        1575672365094440960,
        // @PauloEchebarria
        1576793388845760512,
        // @ProfAlexandreS
        1577470375272878083,
        // @Arnaldodeputado
        1578127734362046489,
        // @Mariada67416358
        1578375608878391296,
        // @TamirisPeixoto1
        1578450870127263757,
        // @CarlosValdevin6
        1579466590512418817,
        // @eusouolimpio
        1580519722877128708,
        // @quinhoprefeito
        1581818536137302021,
        // @marlipaulinopr
        1583101243987206145,
        // @renatodepaularj
        1583183342979190785,
        // @WesleyCosta_GO
        1584241246817624064,
        // @LucasLasmarMG
        1584270172952694784,
        // @Joellobatoc
        1584299141550702593,
        // @andreadantasc
        1584527735686397952,
        // @tcoronel_maia
        1586016154262315009,
        // @rafandradembl
        1586192715137667072,
        // @CombatPatriota
        1586416104632721416,
        // @LucasBackes14pr
        1586733661113794561,
        // @marcosfonsecapi
        1587131451593658369,
        // @antoniodoidoofc
        1588344483766321154,
        // @munique_busson
        1588523690760732672,
        // @RONALDO90199231
        1588685187629678592,
        // @gianninogueira2
        1589336373101723649,
        // @JUNIORCESARLEI9
        1589412695664742401,
        // @LuanNun43052492
        1589645603616817154,
        // @pitypaguiar
        1589741307630690304,
        // @Marisaloboreal
        1589768586096148483,
        // @enzosamuelthe
        1589963570720264193,
        // @soldadoarruda
        1590877665837436928,
        // @ricadeFreitasG1
        1591548255313305607,
        // @MartaPadovani73
        1592147414500122624,
        // @marcelomb1993
        1593005387296514048,
        // @deputadomatheus
        1593032863066251265,
        // @izaarrudape
        1593200340777803776,
        // @davivalencambl
        1594709250835709957,
        // @FelipeBolsonare
        1595399593012887552,
        // @DraLuxmonteiro
        1595451834730168321,
        // @shanvirmond
        1597101228181397508,
        // @Josecar84649656
        1598498909608988673,
        // @DartanhanCampos
        1599722975997067264,
        // @camposoptica
        1599886697096945699,
        // @davipqdtrj
        1600721241534504960,
        // @vandrofamilia
        1603020469145374720,
        // @AlexanderBrasil
        1604310468906176512,
        // @Geraldomendesof
        1605547416366841856,
        // @MarinaCallega13
        1609042267095941123,
        // @gutembergfrios
        1610745043358187520,
        // @WickRyanAM
        1612178912947183620,
        // @SchiavoMaurilio
        1612183554787614722,
        // @deputadairacema
        1612229099551858688,
        // @aveiltonsouza
        1612484459034550273,
        // @MalconMazzucato
        1612520249735135232,
        // @depgilbertinho
        1612589448256016385,
        // @gersonzocchi1
        1612930692932837377,
        // @oGabrielAzevedo
        1612935067487145985,
        // @WaldenorPereira
        1613183981230366728,
        // @DiegoQuaqua
        1613315565203906562,
        // @ToGomes5
        1613522130368356352,
        // @gmoraisdai
        1617357506459598848,
        // @realthiagonunes
        1617967890107432961,
        // @DepLucianoTo
        1618625038176980995,
        // @tadeudesouzaam
        1618970134005121025,
        // @DanielSantanaES
        1619000816714629122,
        // @delegadagabi
        1619448856009138182,
        // @Priscil34438752
        1620207638397976577,
        // @IndiaArmelau
        1622610845002870784,
        // @gracinhamaosant
        1624490961236631553,
        // @francisconace01
        1631055193675644932,
        // @EdsonKambeba
        1633100825403826177,
        // @Jaderbfilho
        1633282883682009091,
        // @PrEzequielbueno
        1633928858180239363,
        // @DeivisOss12
        1634553908402987010,
        // @Dep_CoronelNeil
        1635697153979891712,
        // @LucasAl29111516
        1637240612620587013,
        // @AdrianaLeal2507
        1637963109573828608,
        // @adrianalmeidapt
        1638181881127612416,
        // @leticiaaguiarsp
        1638274776413134849,
        // @Rogerio_Ulysses
        1641171155540123655,
        // @edercostarj
        1642933716279328768,
        // @giuargolo
        1645831168586137600,
        // @fialhooooo
        1646851212329754626,
        // @Edy_Tocantins
        1647755717158264835,
        // @wistongomess
        1648046380106104833,
        // @Alannleal_
        1651297457374887936,
        // @felixsantosrn
        1653188395324014593,
        // @CamilaGodoiSP
        1653466465855471617,
        // @MatiasSamuka
        1655587131136307204,
        // @DeputadoQuirino
        1656674946163277826,
        // @DepComanDANte
        1657817923728166916,
        // @drbrunoresende_
        1661373663256408066,
        // @CosttaIvony
        1662406778263420929,
        // @wellersind
        1663242435420356609,
        // @DouglasRuas_RJ
        1663563614954086401,
        // @bemoreiradf
        1666457401816391682,
        // @TathianaGuzella
        1668266165922144262,
        // @carrarajh23
        1669323757700169729,
        // @vanessarosajlle
        1675490861440802817,
        // @feschmittvet
        1676911232614293505,
        // @GiFreitas1982
        1684192295459905536,
        // @fellipe1971
        1690338078827728896,
        // @babatupinamba_
        1691097420166254592,
        // @tonytretanews
        1691834974301773824,
        // @JacoLulaDaSilva
        1693307464354041856,
        // @souricardojusto
        1693956921449996289,
        // @MarianaNaime
        1700652030501519360,
        // @PauloAssun68233
        1702440269138866176,
        // @mateuspepicepr
        1703957971380604928,
        // @opropriokogos
        1707814589209944064,
        // @rosianepolitica
        1708485431405293568,
        // @marcelodino_rj
        1708844117914972162,
        // @willianrochapr
        1709986969277624320,
        // @osenhorsaldanha
        1710320865517162496,
        // @francisco_arten
        1710430261932908545,
        // @hebertcsgyn
        1713936872370532352,
        // @oengenheiroleo
        1714288040334831616,
        // @TitoBarichello
        1715811403708166144,
        // @karisantospt
        1716415063471316993,
        // @moreiramissao
        1721592893511483392,
        // @ThomazSJC
        1721921193588973568,
        // @jorge6050384252
        1723066872403214336,
        // @guikilter
        1725596039263006720,
        // @Gloria_Vale72
        1725943187250831360,
        // @eduardowilliamm
        1728271638104358912,
        // @ManuellaTyler
        1731034177175171072,
        // @profRodneyRJ
        1731046357425668096,
        // @aureacarolinax
        1734643690751078400,
        // @Aludydias
        1738591794403655682,
        // @MalluAlmeida12
        1739812685187784704,
        // @ysanireal
        1740577335395414017,
        // @drfabriciok
        1742246045835300864,
        // @bellaccarmelo
        1743284668764463105,
        // @bittencourt_rg
        1744512067682316289,
        // @jotinhapiaui
        1744704300616417280,
        // @miriamguzella_
        1750522589548879872,
        // @andrabianchessi
        1752837451175907328,
        // @rmpicoli
        1757491174871363584,
        // @prwellingtonst
        1758502910638329856,
        // @samuel_al_silva
        1760365813456904192,
        // @RicardoAlv32716
        1761085576227213312,
        // @FePresidente
        1762888285796343808,
        // @dulce_lmendes
        1764371188489330689,
        // @JeffreyChiquini
        1767171124629037056,
        // @a_caroliveira
        1767282969708822528,
        // @Lenesilllva
        1771346241449865216,
        // @rafaelsatiebr
        1771625536772571136,
        // @MayaraKeiko
        1773171382886576128,
        // @jorgemaciel05
        1775273264774045697,
        // @gisvaldopsol
        1776773968726298625,
        // @Ap52467Jovelino
        1777297081134178304,
        // @BrunoAlvesSFFS
        1777373652708683786,
        // @fabiocarneirojp
        1777538059891777537,
        // @MarcoRo43166598
        1777608369236271105,
        // @igorrayanrn
        1777621309335183360,
        // @Dhe_paiv
        1777901547629764608,
        // @juhliasantost
        1778077250908274688,
        // @OficialVanucci
        1779252601269141504,
        // @asargentolorena
        1779345802629914624,
        // @carlosiranrs
        1779913435825795072,
        // @BiaCoimbraSP
        1780537023859761153,
        // @PedroNassifrj
        1780805773603258368,
        // @paulacoutinho65
        1781108419853701120,
        // @CamillaGonda
        1781485235047174144,
        // @brunnomattospt
        1782511140561367040,
        // @MicheleMarxsc
        1782916900923543552,
        // @aucilene_a75929
        1784209048671322112,
        // @eliabecamposs
        1784275708606320641,
        // @NivaldoNoga
        1785161935685562368,
        // @oescobarpro
        1786921594390016001,
        // @LeandroGol15952
        1789440263187771392,
        // @GuimaraesAlpha
        1790692252840243201,
        // @EloyNhandewa
        1791120964731744256,
        // @coronellrosses
        1793428861348167680,
        // @akarinaclaro
        1793823661369044992,
        // @nataliademesmao
        1797413322385485825,
        // @Soldado_Sampaio
        1798332723989295104,
        // @Lindenbergbra
        1799431977625366528,
        // @maykondelfinomg
        1800001219932409856,
        // @mqueiroz_rio11
        1800605526683754496,
        // @GilsonMachado22
        1800983500541030400,
        // @severoeulalio
        1801749578938564608,
        // @leorondonn13
        1803570674914426880,
        // @igor_ooliveira1
        1809003112658767872,
        // @gabi_bvnt
        1809421300433317892,
        // @ustramarcelo22
        1809603567520718849,
        // @Adrianadasilvax
        1810502940416991232,
        // @passinhoisa
        1812967972358569984,
        // @TenenteNilton
        1813622960579588096,
        // @RGracie79575
        1814770780871532545,
        // @Fernand51081218
        1814824898956828672,
        // @VolmirGordo
        1815772052063694848,
        // @taniadacreche
        1815835183834390528,
        // @SandroOmar48237
        1816185635578978304,
        // @JoaoRochaFranca
        1816267815063531520,
        // @ofcjusantana
        1820574539312508929,
        // @007Douggomes
        1821931832008392704,
        // @DaSilva69157
        1822302684935790593,
        // @joaogcandidoadv
        1826732949376479232,
        // @betsMartins
        1827352420541763584,
        // @MarcusLopesPsol
        1827593492387745792,
        // @matheussimoespr
        1836487149769596928,
        // @AGoldbach60024
        1844153830201589760,
        // @gabrielpiauhysp
        1844436268706418699,
        // @cironogueirapi
        1844442355421614089,
        // @ledapco
        1845277420854837249,
        // @ProfMarcio58657
        1845590017445400576,
        // @henriquemgjf
        1845839091515981833,
        // @GoulartVla23085
        1846568278950346752,
        // @DragUrbana
        1847431729415397376,
        // @LuizaDoClezao
        1848153540012945408,
        // @abreu_de63729
        1855074530613403648,
        // @luladobemofici
        1855768833115541504,
        // @debora__romani
        1855793756332519424,
        // @LucasReis13_
        1856394921189494784,
        // @sandsonmenezes
        1856816774517268480,
        // @Airtonjose26
        1857173636689362944,
        // @rubensnascimepr
        1857825522765766656,
        // @CristianeN74380
        1858882502707810304,
        // @depcabomacielam
        1859357204278542336,
        // @LusFelipeV68715
        1861380355875373056,
        // @lucylulv
        1863493161240133632,
        // @_AnaHering
        1864125622928187392,
        // @denistaveiradn
        1865917852424798208,
        // @joaopaulo_tprs
        1870514350164705280,
        // @Pcbcastelo
        1872641473021087744,
        // @saulofreitas22
        1873333201445289984,
        // @Lukaovereador
        1874897810753208320,
        // @johnysantos_sp
        1875239088028299264,
        // @oiurecastro
        1878758512639033344,
        // @fsantanapsd55
        1878782691375759360,
        // @FlavioMant5441
        1878922692860006401,
        // @grazimacedo_
        1879635567639777280,
        // @ProfRonaldo13
        1882045370613600257,
        // @_marquinhostrad
        1883938805306052608,
        // @bolsonaro__jr
        1884423273490108416,
        // @digportella
        1884620263549001728,
        // @yasminsarrafsp
        1884758507272216576,
        // @ravengaar
        1885085802645880832,
        // @ameliocayresdep
        1887508211445702656,
        // @FelipeVasquesce
        1889410539639517184,
        // @celprincipebr
        1890456925147734016,
        // @Herculano1011
        1892340577796313092,
        // @julianafideliis
        1896723491371581440,
        // @gualbertoap
        1898078097452548097,
        // @FRomulo13
        1898340153896136704,
        // @Rpachecopinho
        1898684990428151808,
        // @viscontioficial
        1898913625697292288,
        // @GersonClaroMS
        1899113980544897024,
        // @galvaomicheles
        1899797969400180737,
        // @Moanavaladares_
        1899875749907357697,
        // @caixeta_oficial
        1900205421123817472,
        // @falaCosenza
        1900729566542655488,
        // @RafaMinatoSP
        1901680009430892544,
        // @MacAntonioRJ
        1902731327461318656,
        // @LenirOficial
        1904187270342574082,
        // @profeVinicius
        1906160472354873344,
        // @ErikaGuerrieri
        1906823280281174016,
        // @rodrigospada_
        1908175828417949697,
        // @MarinhoGui65411
        1909642092424368128,
        // @glenioseixas
        1909726014202171392,
        // @yolandasilva_sc
        1911181511837011968,
        // @betioldebochado
        1911652608915193856,
        // @DrDiogoFranco
        1915836821826646016,
        // @RenatoAngraRJ
        1916570073663475712,
        // @aparecidobian
        1920469901136764928,
        // @ericafmissao26
        1922067675074727936,
        // @joselaviola
        1922378485412331522,
        // @daniele_carva
        1925558068206518272,
        // @MuriloM50730958
        1928141892065316864,
        // @paulomeloparana
        1929325555234803712,
        // @manoela__peres
        1932149081620484096,
        // @souadibelias
        1932784165545533440,
        // @leo_grandini13
        1932924340917710848,
        // @Camargo1Amanda
        1937005496398962688,
        // @JulianaMontrios
        1937336667356205056,
        // @prof_elson_sc
        1937980838299504644,
        // @marcioalvinosp
        1940410346411597824,
        // @ArturdeFariasV
        1941922310568443904,
        // @beto_vaz_
        1944257271942287360,
        // @jotabrandaoam
        1949664156606459905,
        // @semeadorborges
        1952556730463707137,
        // @DrSergioNeves
        1953095258784579586,
        // @IgorMarquesBRA
        1957556152117653505,
        // @lucasvenditeMS
        1958412111752699904,
        // @NoronhaPro7153
        1958622002719182848,
        // @RubensAngiolett
        1959013340661035008,
        // @sebasticoelho
        1960040602877190144,
        // @isabellaGedeon
        1960414219074998272,
        // @DelAmirSalmen
        1961811802271952896,
        // @blogmmedeiros
        1962570898310840320,
        // @brunoboaretto
        1963656656342192129,
        // @josearrudadf
        1964022075540312064,
        // @juliemilkreal
        1965782186403012608,
        // @delegadali47533
        1966489240566444033,
        // @edilsondamiaorr
        1970177461070528512,
        // @joaoadolfo_14
        1970451742430035968,
        // @Drfilipecm
        1972344363654037504,
        // @delsmontanari
        1973824020907700224,
        // @romulobraz_13
        1975026318975840257,
        // @DrManuelMarcos
        1975317845962858501,
        // @VereadorFox
        1975709711870906369,
        // @CarlosCostaPE10
        1975873882654441472,
        // @RangelJucy18837
        1977431822557700096,
        // @isapaixaosp
        1980463624649945088,
        // @EliCorreaFilhoo
        1980794408141258752,
        // @drjuliostobbe
        1981365553173299200,
        // @Kiko_Caputo
        1982938414002647045,
        // @mazzei_fel47870
        1983280886067195904,
        // @marciaabrahaodf
        1985760866109964288,
        // @alepaivaoficial
        1986993348008419328,
        // @GustavoHenryR
        1987721882838462464,
        // @fig35508
        1988116025670594563,
        // @DaClaudio33805
        1988208205575712772,
        // @RenatoBolsonar0
        1988322911367950337,
        // @vandawitoto
        1988323381792681984,
        // @SamaraMadureira
        1990473255539621889,
        // @YanProf
        1992656375382704128,
        // @lemuel_SV
        1995470250738122752,
        // @MarianaServente
        1996216267678875648,
        // @jorgerrosario
        1997124812662362113,
        // @oleandroissa
        1998066982999212032,
        // @Euberlucas14
        1998418256953233408,
        // @profakellsilva
        1998797409712009216,
        // @joaopiresrj
        2000205916009074690,
        // @nelsongrasselli
        2002006532184281088,
        // @rafoliver_pa
        2005281365072449536,
        // @moisesbarboza
        2006965910309896192,
        // @alvarenga35289
        2007441310102523904,
        // @coronelbusnello
        2009066265156202496,
        // @RogerioChimi
        2010017755194589185,
        // @AntoniadeJessp
        2010525266565857280,
        // @MichelPadao
        2010532868271816705,
        // @profraydf
        2010751427623497728,
        // @PolianadoPsol
        2010869406650245120,
        // @edinhosouzaaa
        2011825328793296896,
        // @Efreu_Quintana
        2013273184640860160,
        // @meunomeejhonebr
        2013398295859609601,
        // @brenomacedopi
        2013660202873032707,
        // @esthermoraessp
        2014421538418642944,
        // @robson_cacau
        2015964533689323520,
        // @carolontiveros0
        2016185817392099336,
        // @anderalvesgo
        2016851339624538112,
        // @Landovoigttt
        2017294158558339072,
        // @ingridcardososp
        2017575066381238272,
        // @sicchar1389
        2018445464908042240,
        // @isabeldesouza01
        2020307564903165953,
        // @vanesckaessusp
        2021065723573829633,
        // @leninhavalente_
        2021229327594000386,
        // @tamirisgodoysp
        2021276913012969472,
        // @MatheusCambuiBa
        2021559145728454656,
        // @ruandutrace
        2021639549625991168,
        // @henriqueduraesd
        2021718062261506049,
        // @Jordambritosc
        2022354011035222016,
        // @sophiafechinece
        2022429125399478273,
        // @mateusquadrosms
        2024092564014669824,
        // @tulioteconta
        2024178238990536704,
        // @NetoFeitos68916
        2026461466728292352,
        // @catarinanevespb
        2026549782278529024,
        // @jessicathis_mbl
        2026754488128909313,
        // @GreguyLoooban
        2027415714953396224,
        // @AraceliLemosOF
        2028870149546373120,
        // @rfurtado22
        2028873608903401472,
        // @mendes__babi
        2029985728059559938,
        // @helen_vitaRJ
        2033587304795963392,
        // @jeaninepresente
        2034019355629846528,
        // @CidaCarvalhoSP
        2034086382705274880,
        // @EdneyBatalha
        2034242236678848513,
        // @WickRyaAM
        2034644862679547904,
        // @Fabio_x86
        2036151026986696704,
        // @MarcaoVivacqua
        2036163287587467265,
        // @isamamedi027
        2036512323020500992,
        // @missaogabi
        2036817223558234112,
        // @marinanamissao
        2036894494973440000,
        // @isabeldesouzasp
        2036944910713081856,
        // @beatrizoPR
        2037168439836512257,
        // @RicardoBec31758
        2037169927618961408,
        // @stellabragasp
        2040479116068081664,
        // @maurodeAL1930
        2041451935685906432,
        // @DepDrFlavio
        2041621533454450688,
        // @oenzozibellini
        2041763427488608256,
        // @Grazypasqualeto
        2042058510536482816,
        // @vivianegomesrn
        2042331395725430784,
        // @CrisNavarroMIDH
        2043108953177899008,
        // @vanessacfortes
        2043516611831701505,
        // @owilsonmartins
        2043714770591735809,
        // @CarolineSa5644
        2043839578394533890,
        // @viniciusdiaspi
        2045135821695574016,
        // @CMolinariBR
        2046074615957540864,
        // @VivianeSan11286
        2046810859695947776,
        // @emersonrrosa
        2047035719500103680,
        // @Barbarabbotega
        2047324463612485632,
        // @JoaoPaulo_2026
        2047395841715920897,
        // @DeyconSP
        2048379106517987328,
        // @VasconcellosCel
        2048607450593468416,
        // @PablodoMST
        2048698490906169344,
        // @CGasparin11011
        2048961988999454720,
        // @livianoronha_PA
        2049127918794670080,
        // @MarcioRezendeRJ
        2049539394701262848,
        // @bia_pedagoga
        2049646600088117249,
        // @enzodangelo_mbl
        2049839952624492544,
        // @oalanmansurrj
        2049927296442658816,
        // @14DanielAguiar
        2050843854325067777,
        // @jaimeverruckms
        2052755712644730887,
        // @Aminjhannouche
        2053565789421047808,
        // @ThefiAmancio21
        2054040940885532672,
        // @flaviadovalmg
        2054361311832616960,
        // @marianasartor50
        2054619154762674176,
        // @onenencoelho
        2054729517864771584,
        // @JulianaBrizola
        2054954070083813376,
        // @mahmoudamer_rs
        2054974955603861504,
        // @luizgamabsb
        2055147146618302464,
        // @DelEduardoK
        2055437449740873728,
        // @glaucelima12
        2056410704031121408,
        // @edmartresoitao
        2056559848750276608,
        // @celmarcioasouza
        2056815030935416832,
        // @CidaVillasBoas
        2056824758201708544,
        // @RodolfoFiorucci
        2057456621865881600,
        // @brenobarcelos_
        2057550738427887616,
        // @PL22Al
        2057808888297062400,
        // @ProfNelsiWelter
        2058186878449278976,
        // @obrasilcomlula
        2058595300873289728,
        // @andreiamoura_df
        2059328849125490688,
        // @eleusespaivaf
        2059680529746644993,
        // @vivianeluizams
        2059726216119107587,
        // @ProfWiterNaves
        2059790203091341316,
        // @profcfabian
        2060000313843572736,
        // @mayaelliz
        2060371890124857347,
        // @cmidf_oficial
        2060549322207375368,
        // @nicolasravipsol
        2061280727711289345,
        // @Renatofonsecape
        2061591502136983552,
        // @DrCrisVeloso
        2061889166779019264,
        // @DaversonMatos
        2062692988606717952,
        // @jmonteirosc
        2062866727252250624,
        // @CoronelMenezes_
        2062867137497059328,
        // @profjoaohs
        2063783489275600896,
        // @viniciusrodsp
        2064052780407336960,
        // @aBiaAlcantara
        2064088166588440577,
        // @DelmaPSOL
        2064731236023541760,
        // @pr_itamar_paim
        2064819106545565696,
        // @MalluCortes
        2065526840198832128,
        // @FelipeGambaroP
        2066731525400330240,
        // @drcassiohprado
        2066997147971497984,
        // @manoelsseverino
        2067256474024173568,
        // @nenemalbuquerqu
        2067271467000045568,
        // @SofiaFavero_
        2067775247542095872,
        // @HVilelaporGoias
        2069828777517973504,
        // @verapinheirosc
        2069869784972369920,
        // @Waltinhogo
        2070185289100775424,
        // @candidotelesdr
        2070216303479066624,
        // @Deborahzanchi_
        2070650915871240192,
        // @Delmariacorsato
        2070667303931256832,
        // @transformarpsol
        2070866245293932545,
        // @avictoriagallo
        2071779601815134208,
        // @angelaabukce
        2071949683736354816,
        // @wilsonamigao
        2071988583334862849,
        // @Drrafujr
        2074177828606652416,
        // @rogeriozabdalla
        2074246655759552513,
        // @MariRibeiro42
        2074876656561393664,
        // @WiliansDouglaas
        2074899795404103681,
        // @drthalescoelho
        2075213887213797376,
        // @ArndaAcademia
        2075242788568899584,
        // @RafaelPerlasca
        2076673278274383872,
        // @AlinemunizRJ
        2076717583307255808,
        // @zealexandrerj
        2076749095092256768,
        // @CarineTAdv
        2077757274961911808,
        // @LarianeTellMend
        2077832017736011777,
        // @DepCarlaMachado
        2078177903271956480,
        // @Nayladasilva0
        2078543961333850112,
        // @drbrenoaraujo
        2079197725384454144,
        // @raimundomce
        2079247986442248192,
        // @evaldo_gomespi
        2079281807556587520,
        // @colet_unidade
        2079306862097215488,
        // @FernandoEsporte
        2079569340517462016,
        // @MaedjaCampos
        2079640052393472000,
        // @PauloMassettiMS
        2079658058842591232,
        // @AlineDinizadv
        2079710939197177856,
        // @PaiRoblez
        2080002260814233600,
        // @miss__Paulinha
        2080010486397988864,
        // @colemulheressp
        2080282215653441537,
        // @helderdelegado
        2080292155315138560,
        // @JoseMoitamxx7
        2080321827289677824,
        // @matteus_hnrq13
        2080378112114593792,
        // @pituxosergipe
        2080630684989730816,
        // @Irmamabelmelo
        2080632627887734784,
        // @thiagocampeloof
        2081154349267345408,
        // @erigreyce__
        2081487029443940352,
        // @DanielSantanamc
        2081774169167974415,
        // @luismartarj
        2081890844735426560,
        // @drjoaomartins26
        2082193801267888129,
        // @Julio_Neto_SP
        2082200815763148800,
        // @BibianoRN
        2082203221980778496,
        // @DorvaniloNilo1
        2082221380297236480,
        // @SubTenSergio
        2082541411111493635,
        // @anamariaa_of
        2082561982377406464,
        // @Fabiani_vasco
        2082567723423342592,
        // @Clarianabr
        2082581908005806081,
        // @joaogusmao_pt
        2082766963659403264,
        // @karimeefayad
        2082897339065233409,
        // @barbararesende0
        2082924803145551872,
        // @guihenriquesc
        2082932165466058752,
        // @odrlucasoficial
        2083260564294295552,
        // @JHONNSOM70
        2083921524394700800,
        // @eliethdefatima
        2084021684495904769,
        // @Profsamuelsiebr
        2084048343609540608,
        // @KelllenGuerra
        2084209199404191745,
        // @profalumatias
        2084258501887406080,
        // @Suely7033
        2084272983443374080,
        // @capitaodaviof
        2084329694938251264,
        // @drgiovanimendes
        2084387783674564608,
        // @tiocarlosrio
        2084426199242022912,
        // @ProfessorBezerr
        2084640378666295296,
        // @AuriJuniorr13
        2084644771927076864,
        // @Rosangelanegaro
        2084724540194643968,
        // @CarmemOliverofc
        2084756136004083712,
        // @PauloLealMissao
        2085083851395620864,
        // @panayotisdolula
        2085133776741437441,
        // @diegojejees
        2085201272139886592,
        // @profyoshida
        2085344308480049152,
        // @lucianoleitoa_
        2085736572914159616,
        // @Virginiaenfer
        2085739235093446656,
        // @profandersonfig
        2085812265509388288,
        // @nandovianapsol
        2086122224126136320,
        // @boracomMarcel
        2086812805034815488,
        // @roneymariachi
        2086873322587881473,
        // @RaphaelaSpadi
        2087175409540497408,
        // @capitaocesario
        2087749550719086592,
        // @DarbideJesusrr
        2088266727519862784,
        // @tatibarrapa
        2088334004445487104,
        // @Mariguedes1406
        2088832528383639552,
        // @PedroAbib15
        2089024802669334528,
        // @brunoscheid222
        2089064009643220993,
        // @RafaGonzagaSP
        2089736144921444352,
        // @marciacandidata
        2089837151844212736,
        // @SimonePimehb
        2090082297370312704,
        // @RadmanGadiel
        2090252697354067968,
        // @cabodaciolo33
        2091177221310345216,
        // @danilosoaresce
        2091948027967676416,
        // @IuriMarque60kt
        2092800629278126081,
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
    /// The Electoral Court order applies to viewers in Brazil, so scope the
    /// filter to them. Without this the filter runs for every viewer in every
    /// country, which is broader than the order requires.
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        query.country_code.eq_ignore_ascii_case(BRAZIL_COUNTRY_CODE)
    }

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
        assert_eq!(BRAZIL_2026_ELECTION_USER_IDS.len(), 2554);
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

    fn query_from_country(country_code: &str) -> ScoredPostsQuery {
        ScoredPostsQuery {
            country_code: country_code.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn enabled_for_viewers_in_brazil() {
        assert!(Brazil2026ElectionFilter.enable(&query_from_country("br")));
    }

    #[test]
    fn enabled_regardless_of_country_code_case() {
        assert!(Brazil2026ElectionFilter.enable(&query_from_country("BR")));
    }

    #[test]
    fn disabled_for_viewers_outside_brazil() {
        for country_code in ["us", "gb", "jp", "pt"] {
            assert!(
                !Brazil2026ElectionFilter.enable(&query_from_country(country_code)),
                "expected filter to be disabled for {country_code}"
            );
        }
    }

    #[test]
    fn disabled_when_country_code_is_unknown() {
        assert!(!Brazil2026ElectionFilter.enable(&ScoredPostsQuery::default()));
    }
}
