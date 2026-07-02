# Lab – Lego Challenge

## Cel projektu

Celem projektu było stworzenie wielomodelowego systemu klasyfikacji kolorów
klocków LEGO na podstawie wartości pikseli RGB oraz ich rozszerzonych reprezentacji
cechowych. System miał:

- klasyfikować kolory LEGO na podstawie 14 cech inżynierii cech,
- porównać klasyczne modele ML z modelami głębokiego uczenia,
- wykorzystać autoenkodery (MLP i CNN) do redukcji wymiarowości,
- zastosować transformer jako model sekwencyjny dla cech,
- umożliwić interaktywną analizę obrazu poprzez GUI i kliknięcia pikseli.

## Zbiór danych

Projekt wykorzystuje dane z plików:

- legocolor-extended.csv – główny zbiór RGB + etykiety kolorów,
- colors.csv – dodatkowe informacje o kolorach (opcjonalne).
Każdy rekord zawiera:
- wartości RGB,
- nazwę koloru LEGO.
Dane zostały oczyszczone (usunięcie braków, konwersja typów) i przygotowane do
dalszej analizy.

Kod:

1. Funkcja load_data()


Wczytywany jest plik legocolor-extended.csv, który znajduje się w folderze archive/. Po
uruchomieniu programu otrzymujemy: "Kształt df_data: (2492, 17), co oznacza 2492
wiersze (próbki) i 17 kolumn. Każdy wiersz odpowiada jednej próbce koloru LEGO.
Parametr delimiter=";" oznacza, że wartości w pliku są rozdzielone średnikami.

2. Funkcja prepare_data()

Funkcja prepare_data() odpowiada za przygotowanie zbioru danych do procesu uczenia
modeli. W pierwszej kolejności sprawdza, czy w pliku znajduje się kolumna „Color”,
zawierająca etykiety klas (nazwy kolorów LEGO). Następnie konwertuje wartości
kanałów R, G i B na dane numeryczne oraz usuwa rekordy zawierające brakujące lub
niepoprawne wartości.

Po oczyszczeniu danych funkcja generuje dla każdego koloru zestaw 14 cech
informacyjnych. Oprócz podstawowych wartości RGB wyznaczane są cechy


pochodzące z innych przestrzeni barw (HSV oraz YUV), znormalizowane składowe RGB
oraz dodatkowe parametry opisujące kolor, takie jak różnica między kanałami
czerwonym i zielonym (Dist_RG) oraz wskaźnik nasycenia (SatIndex).

Wygenerowane cechy są następnie dodawane jako nowe kolumny do zbioru danych.
Na końcu funkcja wyświetla listę dostępnych klas kolorów oraz zestaw
wykorzystywanych cech, po czym zwraca kompletny, oczyszczony i rozszerzony zbiór
danych gotowy do dalszego podziału na zbiór treningowy i testowy oraz do trenowania
modeli uczenia maszynowego.

## Inżynieria cech (14 cech)

Z każdego piksela RGB generowane są 14 cech:

Podstawowe RGB:

- R, G, B

Przestrzenie kolorów:

- Hue, Saturation, Value (HSV)
- Y, U, V (YUV)

Normalizacja:

- R_norm, G_norm, B_norm

Cechy dodatkowe:

- Dist_RG (różnica R-G)
- SatIndex (indeks nasycenia)

Kod:

**1. Normalizacja RGB -> HSV**


RGB to:

- przestrzeń techniczna (kamera)
HSV to:
- przestrzeń ludzka percepcja koloru

_HSV:_

- H (Hue) → kąt na kole barw (0–360°)
- S (Saturation) → czystość koloru
- V (Value) → jasność

Zamiast (R, G, B) otrzymujemy, jaki to jest kolor i jak intensywny

**2. Skalowanie HSV**

HSV z biblioteki jest w zakresie 0- 1

Modele ML lepiej działają na "ludzkich zakresach" (cechy powinny być w podobnych,
małych skalach (np. 0–1 lub standaryzowane), bo modele ML nie radzą sobie z dużymi
różnicami wartości między cechami).

**3. YUV – model światła**

To jest fizyczny model luminancji:

- Zielony ma największy wpływ na jasność
- Czerwony średni
- Niebieski najmniejszy

Y = jasność

U i V opisują odcień koloru i oddzielają kolor od jasności


Model dostaje nie tylko RGB, ale też percepcję (HSV), fizykę światła (YUV),
normalizację, czyli 3 różne perspektywy koloru.

## Normalizacja RGB + cechy dodatkowe

Ten kod zamienia RGB na wersję znormalizowaną, RGB mówi, ile jest światła
czerwonego, zielonego i niebieskiego, ale RGB nie zależy od jasności.

Przykład:

kolor RGB
ciemny czerwony (100, 0, 0)
Jasny czerwony (200, 0, 0)
to ten sam kolor, ale inne wartości

1. Suma kanałów

To mówi, ile łącznie światła jest w pikselu.

2. Podział przed sumę

Zamiast absolutnych wartości jest udział kanałów:

3. Warunek s_rgb == 0

Jeżeli R=0, G=0, B=0 to będzie czarny piksel.

4. DITS_RG


Mierzy różnicę miedzy czerwonym i zielonym, pomaga rozróżniać czerwony vs żółty vs
pomarańczowy.

5. SAT INDEX

0 – szary

1 – bardzo czysty kolor

## Modele klasyczne

Zastosowane modele:

- Logistic Regression
- KNN
- Random Forest
- Gradient Boosting
- SVC (RBF)
- Linear SVC
- Perceptron
- MLP
- Gaussian Naive Bayes

Metoda oceny:

- 5 - fold cross validation
- metryka: accuracy

Wynik:

Najlepszy model wybierany jest automatycznie na podstawie średniej CV accuracy.

Kod

1. Funkcja train_and_select_classical_models


Funkcja odpowiada za kompleksowy proces trenowania oraz porównania klasycznych
modeli uczenia maszynowego, a następnie wybór najlepszego z nich.

Na początku dane wejściowe (X_train) są standaryzowane przy użyciu
StandardScaler, co zapewnia ujednolicenie skali cech i poprawia stabilność wielu
algorytmów ML.

Następnie definiowany jest zestaw klasycznych modeli uczenia maszynowego,
obejmujący algorytmy liniowe, probabilistyczne, oparte o najbliższych sąsiadów,
maszyny wektorów nośnych, modele drzewiaste oraz prostą sieć neuronową.

Dla każdego modelu przeprowadzana jest 5-krotna walidacja krzyżowa
(cross_val_score) z metryką accuracy, co pozwala ocenić jego zdolność do
generalizacji. Wyniki są zapisywane i porównywane.


Po zakończeniu oceny wszystkich modeli wybierany jest najlepszy model na podstawie
średniej dokładności z walidacji krzyżowej. Model ten jest następnie trenowany na
całym zbiorze treningowym.

Funkcja zwraca:

- najlepszy model,
- obiekt skalera,
- listę wyników wszystkich modeli,
- przeskalowany zbiór treningowy.
2. Funkcja evaluate_model

Funkcja służy do oceny finalnego, wybranego modelu klasycznego na zbiorze
testowym.

Na początku dane testowe są przekształcane przy użyciu wcześniej dopasowanego
skalera, aby zachować spójność transformacji z danymi treningowymi.

Następnie model generuje predykcje dla zbioru testowego. Na ich podstawie obliczane
są podstawowe metryki jakości klasyfikacji: accuracy, precision, recall oraz F1-score,
przy użyciu średniej ważonej.


Dodatkowo generowany jest szczegółowy raport klasyfikacji (classification_report),
który przedstawia jakość predykcji dla każdej klasy osobno.

Funkcja tworzy również macierz pomyłek (confusion matrix), która jest wizualizowana
za pomocą mapy cieplnej, co umożliwia analizę błędów klasyfikatora.

Na końcu funkcja zwraca obliczone metryki oraz predykcje modelu.

3. Funkcja plot_feature_importance

Funkcja służy do wizualizacji ważności cech wejściowych w modelu, o ile dany model
udostępnia atrybut feature_importances_.

Jeżeli model posiada ten atrybut (co jest typowe dla modeli drzewiastych, takich jak
Random Forest czy Gradient Boosting), funkcja pobiera wartości ważności cech i
przedstawia je w formie wykresu słupkowego.

Wykres umożliwia interpretację wpływu poszczególnych cech na decyzje modelu.

Jeżeli model nie udostępnia informacji o ważności cech, funkcja informuje o braku
możliwości wygenerowania wykresu i kończy działanie.

Podsumowanie

Cały fragment kodu realizuje kompletny pipeline klasycznego uczenia maszynowego:

- train_and_select_classical_models → porównanie i wybór modelu
- evaluate_model → ocena jakości najlepszego modelu
- plot_feature_importance → interpretacja modelu (jeśli możliwa)

## Przygotowanie danych dla PyTorch

W projekcie dane początkowo znajdują się w postaci tablic NumPy: x_train _,_ x_test oraz
etykiet y_train _,_ y_test. Natomiast modele PyTorch nie pracują bezpośrednio na
tablicach NumPy, ale na obiektach typu: torch.Tensor, dlatego jest potrzebna klasa,


która zamieni dane do odpowiedniego formatu i umożliwi ich wygodne pobieranie
podczas treningu.

Kod

1. Definicja klasy

Klasa dziedziczy po Dataset z modułu torch.utils.data. Oznacza to, że staje się własnym
zbiorem danych zgodnym z wymaganiami PyTorch.

2. Konstruktor __init__

Konstruktor jest wywoływany podczas tworzenia obiektu klasy. Przykład:

```
Dataset = FeatureDataset(x_train , y_train)
```
Parametry:

- x – macierz cech wejściowych
- y – etykiety klas (opcjonalne)
3. Konwersja danych do tensorów

Dane wejściowe są zamieniane na tensor PyTorch typu float32. Typ float32 jest
standardowo wykorzystywany podczas obliczeń w sieciach neuronowych.

4. Konwersja etykiet klas

Tutaj sprawdzane jest, czy przekazano etykiety. Jeżeli y is None to self.y = None,
przypadek taki występuje np. podczas kodowania danych przez autoenkoder, gdzie
potrzebne są jedynie cechy wejściowe. Jeżeli etykiety istnieją (y = [0, 2, 1, 5]) to zostaną
zamienione na tensor (tensor([0, 2, 1, 5])) o typie torch.long, typ long jest wymagany
przez funkcję CrossEntropyLoss() wykorzstywaną w klasyfikacji.

5. Metoda __len__


Zwraca liczbę próbek znajdujących się w zbiorze danych.

6. Metoda __getitem__

Metoda odpowiada za pobranie pojedynczej próbki o indeksie idx.

**Jak wykorzystywana jest ta klasa?**

FeatureDataset(X_train_scaled) tworzy dataset – tylko cechy X, bez etykiet y, ponieważ
enkoder uczy się rekonstrukcji.

Tworzony jest DataLoader, który dzieli dane na batch'e i losuje kolejność (shuffle=True).

W treningu PyTorch automatycznie wywołuje __getitem __, który pobiera kolejne próbki i
tworzy batch

Schemat działania


## Autoenkoder MLP (14 cech)

Zastosowano sieć:

- Encoder: 14 → 64 → latent (8)
- Decoder: latent → 64 → 14

Cel:

- redukcja wymiarowości,
- wyuczenie ukrytej reprezentacji koloru.

Następnie:

- latent → RandomForest → klasyfikacja

Kod

1. Klasa Autoencoder


Klasa Autoencoder definiuje prosty autoenkoder oparty na sieci neuronowej MLP,
którego zadaniem jest nauczenie się kompresji i rekonstrukcji danych wejściowych.
Model składa się z dwóch głównych części: enkodera oraz dekodera. Enkoder przyjmuje
dane o wymiarze input_dim (np. 14 cech opisujących kolory LEGO) i przekształca je
najpierw do przestrzeni o większej liczbie neuronów (64), a następnie redukuje ich
wymiar do postaci latentnej latent_dim, tworząc skompresowaną reprezentację
danych. W tym etapie wykorzystywana jest nieliniowa funkcja aktywacji ReLU, która
pozwala modelowi uczyć się bardziej złożonych zależności między cechami. Następnie
dekoder wykonuje proces odwrotny — przekształca wektor latentny z powrotem do
przestrzeni o wymiarze oryginalnych danych, próbując odtworzyć wejście. W metodzie
forward dane wejściowe przechodzą przez enkoder, gdzie powstaje wektor latentny z,
który następnie jest przekazywany do dekodera w celu rekonstrukcji x_rec. Model
zwraca zarówno odtworzone dane, jak i reprezentację latentną, co umożliwia
wykorzystanie go zarówno do rekonstrukcji, jak i redukcji wymiarowości danych.

2. Funkcja train_autoencoder


Funkcja train_autoencoder odpowiada za pełny proces uczenia autoenkodera MLP
na przygotowanych 14 cechach opisujących kolory LEGO. Na początku dane wejściowe
X_train_scaled są opakowywane w klasę FeatureDataset, która umożliwia ich
obsługę przez PyTorch, a następnie przekazywane do DataLoader, który dzieli je na
mini-batche i losowo miesza dane w trakcie treningu. Następnie tworzony jest model
autoenkodera o zadanym wymiarze wejściowym input_dim i wymiarze latentnym
latent_dim, który przenoszony jest na wybrane urządzenie obliczeniowe (CPU/GPU).
Do uczenia modelu wykorzystywana jest funkcja straty MSE (Mean Squared Error), która
mierzy różnicę pomiędzy danymi wejściowymi a ich rekonstrukcją, oraz optymalizator
Adam, odpowiedzialny za aktualizację wag sieci na podstawie gradientów błędu.

Proces uczenia przebiega w pętli po epokach, gdzie dla każdej epoki model przetwarza
kolejne batch’e danych: wykonuje propagację w przód, generując rekonstrukcję
wejścia, oblicza błąd rekonstrukcji, a następnie wykonuje propagację wsteczną w celu
aktualizacji wag. W trakcie każdej epoki akumulowany jest całkowity błąd, który
następnie jest uśredniany i wypisywany jako informacja o postępie uczenia. Dzięki
temu możliwe jest monitorowanie procesu uczenia i ocena, czy model poprawnie uczy
się kompresji danych. Po zakończeniu treningu funkcja zwraca wytrenowany model
autoenkodera, który może być wykorzystany do ekstrakcji reprezentacji latentnych lub
rekonstrukcji danych wejściowych.

3. encode_with ___ autoencoder


Funkcja encode_with_autoencoder służy do przekształcenia danych wejściowych do
przestrzeni latentnej (ukrytej reprezentacji) przy użyciu wcześniej wytrenowanego
autoenkodera MLP. Na początku model przełączany jest w tryb ewaluacji
(model.eval()), co wyłącza mechanizmy treningowe takie jak dropout i zapewnia
stabilne działanie sieci. Następnie dane X_scaled są opakowywane w klasę
FeatureDataset, a potem przekazywane do DataLoader, który dzieli je na batch’e (w tym
przypadku po 256 próbek) bez losowego mieszania, ponieważ zachowanie kolejności
danych jest istotne podczas kodowania.

W kolejnym kroku funkcja przechodzi przez wszystkie batch’e danych w trybie
torch.no_grad(), co oznacza, że PyTorch nie zapisuje gradientów, ponieważ nie jest
wykonywany trening, a jedynie inferencja. Dla każdego batcha dane są przenoszone na
odpowiednie urządzenie obliczeniowe, a następnie przepuszczane przez autoenkoder.
Z modelu pobierana jest jedynie część latentna z, czyli skompresowana reprezentacja
danych, natomiast rekonstrukcja wejścia jest ignorowana. Otrzymane wektory latentne
są konwertowane do formatu NumPy i dodawane do listy wyników.

Na końcu wszystkie batch’e są łączone pionowo funkcją np.vstack, tworząc jedną
macierz Z, która zawiera latentne reprezentacje wszystkich próbek. Wynikowa macierz
stanowi nową, zredukowaną reprezentację danych, która może być dalej wykorzystana
np. do klasyfikacji (np. Random Forest) lub wizualizacji.

## Transformer dla cech

Zastosowano prosty transformer:

- projekcja cech do d_model = 64
- encoder transformerowy
- global average pooling
- klasyfikacja softmax


Traktowanie 14 cech jako „sekwencji cech”, a nie wektora statycznego.

Kod

1. Klasa FeatureTransformerClassifier

To klasa definiująca model klasyfikacyjny oparty na architekturze Transformer,
zaimplementowany w PyTorch.

Model składa się z trzech głównych części:

1. Projekcja wejścia (self.proj)
Warstwa liniowa (nn.Linear), która przekształca wejściowy wektor cech o wymiarze
input_dim do przestrzeni ukrytej o wymiarze d_model. Dzięki temu dane są
dopasowane do wymagań transformera.
2. Encoder Transformer (self.encoder)
Właściwa część modelu oparta na nn.TransformerEncoder. Składa się z num_layers
warstw enkodera, z których każda zawiera mechanizm multi-head self-attention
(nhead) oraz sieć feed-forward (dim_feedforward).
Model analizuje zależności między cechami w reprezentacji sekwencyjnej.
3. Warstwa klasyfikacyjna (self.cls_head)
Warstwa liniowa mapująca reprezentację ukrytą (d_model) na liczbę klas
(num_classes).

Forward pass

W metodzie forward:

1. Dane wejściowe są rzutowane do przestrzeni d_model.


2. Dodawany jest wymiar sekwencji (unsqueeze(1)), traktując cały wektor cech
    jako sekwencję długości 1.
3. Dane przechodzą przez encoder Transformer.
4. Następuje uśrednienie po wymiarze sekwencji.
5. Wynik trafia do warstwy klasyfikacyjnej, która zwraca logity.
2. Funkcja train_transformer_classifier

Funkcja odpowiada za trenowanie modelu Transformerowego.

Działanie krok po kroku:

1. Tworzy obiekt FeatureDataset z danych treningowych.
2. Opakowuje dane w DataLoader, który umożliwia batchowanie i losowe
    mieszanie danych.
3. Inicjalizuje model FeatureTransformerClassifier i przenosi go na urządzenie
    (DEVICE).
4. Definiuje:
    a. funkcję straty: CrossEntropyLoss (dla klasyfikacji wieloklasowej),
    b. optymalizator: Adam.


Pętla treningowa:

Dla każdej epoki:

- model przetwarza batch danych,
- obliczana jest strata,
- wykonywany jest backpropagation,
- aktualizowane są wagi,
- zbierane są statystyki (loss i accuracy).
Na końcu każdej epoki wypisywane są:
- średnia strata,
- dokładność klasyfikacji.
Funkcja zwraca wytrenowany model.
3. Funkcja evaluate_transformer

Funkcja służy do oceny wytrenowanego modelu na zbiorze testowym.

Działanie:

1. Ustawia model w tryb ewaluacji (model.eval()).
2. Tworzy dataset i DataLoader dla danych testowych.


3. Przechodzi przez dane bez gradientów (torch.no_grad()), aby:
    a. wygenerować predykcje,
    b. zebrać etykiety rzeczywiste.
4. Łączy wyniki z batchy w pełne tablice NumPy.

Metryki:

Obliczane są standardowe miary jakości klasyfikacji:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
Na końcu wyniki są wypisywane i zwracane.

## CNN Autoencoder (11x11 patch)

Dane wejściowe:

- wycinki obrazu 11×11 pikseli

Model:

- encoder CNN (Conv + MaxPool)
- latent vector (8D)
- decoder (ConvTranspose)

Cel:

- uwzględnienie tekstury plastiku LEGO
- odporność na szum i oświetlenie

Dodatkowo:

Generowane są syntetyczne patche z:

- gradientem światła
- szumem aparatu

Kod

1. Funkcja extract_real_patch


Funkcja służy do bezpiecznego wycinania fragmentów obrazu o stałym rozmiarze 11×
pikseli wokół wskazanego punktu (x, y). Najpierw wyznaczany jest obszar wycinka z
uwzględnieniem granic obrazu, aby uniknąć wyjścia poza jego zakres. Następnie
pobierany jest odpowiedni fragment obrazu. W przypadku gdy wycięty patch ma zbyt
mały rozmiar (np. przy krawędziach obrazu), brakujące piksele są uzupełniane zerami
poprzez padding, tak aby zawsze uzyskać spójny rozmiar wejścia. Na końcu patch jest
przekształcany do formatu kanałów (C, H, W) oraz normalizowany do zakresu [0, 1], co
przygotowuje go do dalszego przetwarzania przez sieć neuronową.

2. Funkcja generate_textured_patch


Funkcja generuje sztuczny patch obrazu o zadanych wartościach RGB, wzbogacony o
elementy imitujące rzeczywiste właściwości wizualne. Na początku tworzony jest
jednolity patch wypełniony kolorem określonym przez wartości (r, g, b). Następnie
dodawany jest delikatny gradient przestrzenny, który symuluje zmiany oświetlenia i
odbicia światła na powierzchni. Dodatkowo wprowadzany jest losowy szum
Gaussowski, który imituje zakłócenia charakterystyczne dla sensorów kamer. Na końcu
wartości są ograniczane do zakresu [0, 1], aby zachować poprawność danych
wejściowych.

3. Klasa PatchDataset

Klasa PatchDataset stanowi implementację zbioru danych w formacie zgodnym z
PyTorch i przechowuje zestaw patchy obrazów. Jej zadaniem jest umożliwienie
wygodnego dostępu do danych w procesie uczenia sieci neuronowej. Metoda __len__
zwraca liczbę dostępnych próbek, natomiast __getitem__ zwraca pojedynczy patch w
postaci tensora typu float32. Dzięki tej strukturze dane mogą być efektywnie
przetwarzane przez mechanizm DataLoader.


4. Klasa CNNAutoencoder

Klasa CNNAutoencoder definiuje konwolucyjny autoenkoder przeznaczony do
kompresji i rekonstrukcji małych fragmentów obrazu 11×11. Model składa się z
enkodera oraz dekodera. Enkoder wykorzystuje warstwy konwolucyjne i operacje max
pooling do stopniowego zmniejszania wymiaru przestrzennego danych z jednoczesnym


zwiększaniem liczby kanałów, aż do uzyskania zwartej reprezentacji latentnej o
zadanym wymiarze. Następnie dane są spłaszczane i mapowane do wektora latentnego
za pomocą warstwy liniowej. Dekoder wykonuje proces odwrotny, przekształcając
wektor latentny z powrotem do przestrzeni obrazowej przy użyciu warstw transposed
convolution, stopniowo zwiększając rozdzielczość aż do oryginalnego rozmiaru 11×11.
Funkcja aktywacji sigmoid zapewnia, że wartości wyjściowe mieszczą się w zakresie [0,
1]. Metody encode, decode oraz forward odpowiadają odpowiednio za kodowanie,
dekodowanie oraz pełny przepływ danych przez model.

5. Funkcja train_cnn_autoencoder

Funkcja odpowiada za proces trenowania konwolucyjnego autoenkodera na zbiorze
patchy obrazów. Na początku tworzony jest dataset oraz DataLoader, które umożliwiają
przetwarzanie danych w batchach. Następnie inicjalizowany jest model autoenkodera,
funkcja straty MSE oraz optymalizator Adam. W trakcie uczenia model przetwarza
kolejne partie danych, generując rekonstrukcje wejściowych patchy, a następnie
obliczany jest błąd rekonstrukcji pomiędzy wejściem a wyjściem. Na jego podstawie
wykonywana jest propagacja wsteczna i aktualizacja wag. Po każdej epoce
raportowana jest średnia wartość straty, co pozwala monitorować proces uczenia.

6. Funkcja encode_with_cnn_autoencoder


Funkcja służy do ekstrakcji reprezentacji latentnych dla zbioru patchy przy użyciu
wytrenowanego autoenkodera. Model ustawiany jest w tryb ewaluacji, a dane
przetwarzane są bez obliczania gradientów. Każdy batch patchy jest kodowany do
wektora latentnego, który następnie jest zapisywany i łączony w jedną macierz.
Wynikiem działania funkcji jest macierz reprezentacji latentnych, która może być
wykorzystywana do dalszej analizy, klasyfikacji lub klasteryzacji danych.

## GUI – interaktywny system

System zawiera GUI oparte o Tkinter:

Funkcje:

- wybór obrazu,
- kliknięcie piksela,
- predykcja koloru:

Modele w GUI:

- klasyczny model 14F,
- CNN latent + RandomForest.

Kod

1. Funkcja ope_image_and_click


Funkcja umożliwia interaktywną analizę obrazu poprzez kliknięcie w wybrany piksel i
predykcję jego klasy na podstawie dwóch różnych podejść modelowych. Na początku
obraz jest wczytywany z podanej ścieżki, konwertowany do przestrzeni RGB, a
następnie przekształcany do tablicy NumPy. Obraz jest wyświetlany w oknie
graficznym, które pozwala użytkownikowi na wskazywanie punktów za pomocą
kliknięcia myszy.

Po kliknięciu w dowolny piksel pobierane są jego współrzędne (x, y) oraz wartości
składowych koloru RGB. Na tej podstawie tworzony jest wektor cech wejściowych dla
klasycznego modelu ML (14 cech), który jest następnie skalowany przy użyciu wcześniej
dopasowanego skalera i przekazywany do modelu klasyfikacyjnego. Model zwraca
predykcję klasy, która jest następnie zamieniana z kodu liczbowego na nazwę koloru
przy użyciu LabelEncoder.


Dodatkowo, jeżeli dostępny jest model CNN wraz z klasyfikatorem działającym na
przestrzeni latentnej, funkcja wykonuje analizę z wykorzystaniem lokalnego kontekstu
przestrzennego. W tym celu z obrazu wycinany jest patch o stałym rozmiarze wokół
klikniętego punktu, który następnie jest przetwarzany przez autoenkoder CNN w celu
uzyskania reprezentacji latentnej. Uzyskany wektor jest skalowany i przekazywany do
klasyfikatora Random Forest działającego w przestrzeni latentnej, który również
przewiduje klasę koloru.

Na końcu interfejs wizualny oznacza kliknięty punkt na obrazie czerwonym
znacznikiem, co pozwala użytkownikowi śledzić wykonane predykcje bezpośrednio na
obrazie.

## Pipeline główny

W main() wykonywane są kolejno:

1. Wczytanie danych
2. Feature engineering (14 cech)
3. Klasyczne modele + wybór najlepszego
4. Autoencoder + RF
5. Transformer
6. CNN autoencoder + RF
7. Ewaluacja wszystkich modeli
8. GUI + test obrazu

Schemat całego procesu:


## Wyniki

1. Wizualizacja przestrzeni RGB


Na rysunku przedstawiono rozkład kolorów LEGO w przestrzeni RGB. Widać wyraźne
skupiska odpowiadające poszczególnym barwom. Niektóre kolory (np. Orange, Red,
Bright Pink) są dobrze odseparowane, natomiast inne (Dark Bluish Grey, Light Bluish
Grey, Black) tworzą nakładające się klastry, co sugeruje trudność klasyfikacji.

2. Klasyczne modele ML – wyniki


Przetestowano 9 klasycznych modeli ML. Najwyższe CV Accuracy uzyskał
MLPClassifier (0.8655), a następnie SVC RBF (0.8605) i RandomForest (0.8565).

Na zbiorze testowym najlepszy model (MLP) osiągnął:

- Accuracy: 0.8377
- Precision: 0.8379
- Recall: 0.8377
- F1-score: 0.8373
Macierz pomyłek pokazuje, że model bardzo dobrze klasyfikuje kolory takie jak Orange,
Red, Bright Pink, Medium Azure, natomiast największe problemy występują dla:
- Dark Bluish Grey
- Light Bluish Grey
- Black
Są to kolory o bardzo podobnych wartościach RGB.
3. Autoencoder MLP – wyniki


Autoencoder MLP kompresował 14 cech do 8-wymiarowego wektora latentnego.
Następnie trenowano RandomForest na latentach.

Wyniki:

- Accuracy: 0.8317
- Precision: 0.8301
- Recall: 0.8317
- F1-score: 0.8299
Wyniki są bardzo zbliżone do klasycznego modelu, co oznacza, że autoencoder nauczył
się sensownej reprezentacji, ale nie poprawił jakości klasyfikacji.
4. Transformer – wyniki

Transformer traktował wektor 14 cech jako sekwencję długości 1 i uczył się zależności
między cechami za pomocą mechanizmu uwagi.

Wyniki:

- Accuracy: 0.8056
- Precision: 0.8158
- Recall: 0.8056
- F1-score: 0.8080
Transformer działa poprawnie, ale nie przewyższył klasycznych modeli.
5. CNN Autoencoder 11x11 – wyniki

CNN autoencoder uczył się reprezentacji koloru na patchach 11×11.

Wyniki klasyfikacji RF na latentach CNN:

- Accuracy: 0.6593
- Precision: 0.6709
- Recall: 0.6593
- F1-score: 0.6585
Wynik jest niższy, ponieważ patch był jednolity (brak tekstury), a CNN nie miało czego
się nauczyć poza kolorem — co klasyczne cechy robią lepiej.
6. GUI - działanie systemu


Zaimplementowano graficzny interfejs użytkownika umożliwiający wczytanie zdjęcia i
kliknięcie w dowolny piksel.

GUI wyświetla trzy predykcje:

- klasyczny model (14 cech)
- CNN autoencoder + RF
- transformer
Na przykładzie widać, że klasyczny model przewidział kolor jako Black, natomiast CNN
jako Dark Bluish Grey, co pokazuje różnice w podejściu modeli.

## Podsumowanie

W projekcie opracowano kompletny system klasyfikacji kolorów klocków LEGO na
podstawie pojedynczego piksela obrazu. Dane wejściowe zostały rozszerzone z


surowych wartości RGB do **14 cech informacyjnych** , obejmujących przestrzenie HSV,
YUV, znormalizowane RGB oraz dodatkowe wskaźniki (Dist_RG, SatIndex). Analiza
wizualna w przestrzeni RGB wykazała, że część kolorów tworzy dobrze odseparowane
skupiska (np. Orange, Red, Bright Pink), natomiast inne — szczególnie odcienie
szarości i czerń — silnie na siebie nachodzą, co utrudnia klasyfikację.

Przetestowano szeroki zestaw modeli: klasyczne algorytmy ML, autoencoder MLP,
transformer oraz CNN autoencoder operujący na patchach 11×11. Najlepsze wyniki
uzyskał **MLPClassifier** trenowany na 14 cechach, osiągając **Accuracy ≈ 0.84** na zbiorze
testowym. Autoencoder MLP z klasyfikatorem RF uzyskał wynik bardzo zbliżony (≈0.83),
natomiast transformer osiągnął około 0.81. CNN autoencoder, mimo bardzo niskiego
błędu rekonstrukcji, uzyskał najniższy wynik klasyfikacji (≈0.66), co wynika z faktu, że
jednolite patche nie zawierają informacji przestrzennych, które mogłyby dać przewagę
sieci konwolucyjnej.

Zaimplementowano również interaktywne GUI umożliwiające wczytanie zdjęcia i
kliknięcie w dowolny piksel, po czym system prezentuje trzy niezależne predykcje:
klasycznego modelu, modelu CNN oraz transformera. Pozwala to użytkownikowi
porównać zachowanie różnych architektur w praktycznym zastosowaniu.

## Wnioski

- Cechy informacyjne są kluczowe. Zastosowanie 14 cech znacząco poprawia
    jakość klasyfikacji w porównaniu z surowym RGB. Modele klasyczne
    wykorzystują te cechy bardzo efektywnie.
- Klasyczne modele ML przewyższają modele głębokie w tym zadaniu. Problem
    klasyfikacji koloru jest stosunkowo prosty i dobrze opisany przez cechy
    tablicowe, dlatego MLPClassifier i SVC osiągają najlepsze wyniki.
- Autoencoder MLP uczy się sensownej reprezentacji, ale nie poprawia wyników.
    Latenty AE są jakościowe, jednak nie przewyższają ręcznie zaprojektowanych
    cech.
- Transformer nie ma przewagi przy danych bez struktury sekwencyjnej. Wektor 14
    cech nie stanowi sekwencji, dlatego mechanizm uwagi nie wnosi dodatkowej
    informacji.
- CNN autoencoder nie sprawdza się na jednolitych patchach. CNN potrzebuje
    tekstury lub struktury przestrzennej, a jednolite 11×11 pola nie zawierają takich
    informacji. Stąd najniższy wynik klasyfikacji.
- Najtrudniejsze kolory do klasyfikacji to odcienie szarości i czerń. Dark Bluish
    Grey, Light Bluish Grey i Black mają bardzo zbliżone wartości RGB, co prowadzi
    do częstszych pomyłek — potwierdza to macierz pomyłek.


- GUI potwierdza praktyczną użyteczność systemu. Interfejs umożliwia szybkie
    testowanie modeli na realnych zdjęciach, co pokazuje różnice w zachowaniu
    poszczególnych architektur.

