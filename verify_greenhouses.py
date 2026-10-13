import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
from shapely.geometry import Polygon
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from imblearn.over_sampling import SMOTE

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=UserWarning, message="Column names longer than 10 characters")

# === CONFIGURAÇÃO INICIAL ===

poligonos_path = r"path/to/greenhouses.gdb"      # registry (GDB or shapefile); field "Confirmado" = ground truth (1/0)
poligonos_layer = "greenhouse_polygons"

raster_path = r"path/to/rgb_orthoimagery.tif"    # high-resolution RGB orthoimagery (e.g. DGT ortoSat2023)

output_gdb = r"path/to/output.gdb"

output_layer = "estufas_confirmadas"

output_dir = r"path/to/output"

results_dir = os.path.join(
    output_dir,
    "Resultados_confirma_estufas_SIA"
)

os.makedirs(results_dir, exist_ok=True)

# Parâmetros ajustáveis
buffer_metros = 0  # Buffer em metros (0 = usar exatamente o polígono)
num_cv_folds = 5
test_size = 0.25

print("📁 A ler dados...")
gdf = gpd.read_file(poligonos_path, layer=poligonos_layer)
raster = rasterio.open(raster_path)

# Verificação de dados de entrada
print(f"CRS do shapefile: {gdf.crs}")
print(f"CRS do raster: {raster.crs}")
print(f"Número de bandas no raster: {raster.count}")
print(f"Descrições das bandas: {raster.descriptions}")

if gdf.crs != raster.crs:
    print("⚠️ AVISO: Os sistemas de coordenadas não coincidem!")
    print("A reprojetar shapefile para coincidir com o raster...")
    gdf = gdf.to_crs(raster.crs)

# Verificar se existe o campo 'Conf_novo'
if 'Conf_novo' not in gdf.columns:
    print("⚠️ AVISO: O shapefile não contém o campo 'Conf_novo'")
    print("A adicionar coluna 'Conf_novo' com valor padrão 1...")
    gdf['Conf_novo'] = 1

print(f"Total de polígonos no shapefile: {len(gdf)}")
print(f"Polígonos confirmados: {gdf['Conf_novo'].value_counts().to_dict()}")

def extrair_estatisticas_poligono_rgb(geom, src, buffer=0):
    """Extrai estatísticas RGB de um polígono a partir de um raster"""
    try:
        # Aplicar buffer se solicitado
        if buffer > 0:
            geom = geom.buffer(buffer)
        
        # Verificar se a geometria é válida
        if not geom.is_valid or geom is None:
            return None
        
        # Recortar o raster usando o polígono como máscara
        out_image, out_transform = rasterio.mask.mask(src, [geom], crop=True, all_touched=False)
        
        # Usar apenas as 3 primeiras bandas (RGB)
        out_image = out_image[:3]  # Garantir que usamos apenas RGB
        
        # Substituir valores nodata por NaN
        if src.nodata is not None:
            out_image = np.where(out_image == src.nodata, np.nan, out_image)
        
        # Verificar se há dados válidos suficientes (pelo menos 50% não-NaN)
        valid_percent = np.sum(~np.isnan(out_image)) / out_image.size
        if valid_percent < 0.5:
            return None
        
        # Calcular estatísticas por banda RGB
        stats = {}
        
        # Estatísticas básicas
        red = out_image[0].astype(float)
        green = out_image[1].astype(float)
        blue = out_image[2].astype(float)
        
        stats['media_R'] = np.nanmean(red)
        stats['media_G'] = np.nanmean(green)
        stats['media_B'] = np.nanmean(blue)
        
        stats['std_R'] = np.nanstd(red)
        stats['std_G'] = np.nanstd(green)
        stats['std_B'] = np.nanstd(blue)
        
        # Percentis
        stats['p25_R'] = np.nanpercentile(red, 25)
        stats['p75_R'] = np.nanpercentile(red, 75)
        stats['p25_G'] = np.nanpercentile(green, 25)
        stats['p75_G'] = np.nanpercentile(green, 75)
        stats['p25_B'] = np.nanpercentile(blue, 25)
        stats['p75_B'] = np.nanpercentile(blue, 75)
        
        # Textura (variação local)
        for banda_idx, (banda, nome) in enumerate(zip([red, green, blue], ['R', 'G', 'B'])):
            if banda.size <= 1:
                stats[f'textura_{nome}'] = np.nan
            else:
                # Diferença horizontal
                diff_h = np.abs(np.diff(banda, axis=1))
                # Diferença vertical
                diff_v = np.abs(np.diff(banda, axis=0))
                # Média das diferenças
                textura_valor = np.nanmean(np.concatenate([diff_h.flatten(), diff_v.flatten()]))
                stats[f'textura_{nome}'] = textura_valor
        
        # Índices de cor RGB
        # Índice de Vegetação por Diferença Normalizada de Verde (GNDVI alternativo)
        stats['gr_index'] = np.nanmean((green - red) / (green + red + 1e-10))
        stats['gb_index'] = np.nanmean((green - blue) / (green + blue + 1e-10))
        stats['rb_index'] = np.nanmean((red - blue) / (red + blue + 1e-10))
        
        # Índice de Vegetação de Excesso de Verde (ExG)
        r_norm = red / (red + green + blue + 1e-10)
        g_norm = green / (red + green + blue + 1e-10)
        b_norm = blue / (red + green + blue + 1e-10)
        stats['exg'] = np.nanmean(2 * g_norm - r_norm - b_norm)
        
        # Índice de Vegetação de Excesso de Verde menos Excesso de Vermelho (ExGR)
        exr = np.nanmean(1.4 * r_norm - g_norm)
        stats['exgr'] = stats['exg'] - exr
        
        # Índice de Área de Vegetação (VARI)
        stats['vari'] = np.nanmean((green - red) / (green + red - blue + 1e-10))
        
        # Brilho médio (média das três bandas)
        stats['brightness'] = np.nanmean((red + green + blue) / 3)
        
        # Saturação (diferença entre max e min)
        rgb_stack = np.stack([red, green, blue], axis=0)
        stats['saturation'] = np.nanmean(np.nanmax(rgb_stack, axis=0) - np.nanmin(rgb_stack, axis=0))
        
        # Hue dominante simplificado
        max_channel = np.argmax([stats['media_R'], stats['media_G'], stats['media_B']])
        stats['dominant_channel'] = max_channel  # 0=R, 1=G, 2=B
        
        # Ratios entre bandas
        stats['r_g_ratio'] = stats['media_R'] / (stats['media_G'] + 1e-10)
        stats['r_b_ratio'] = stats['media_R'] / (stats['media_B'] + 1e-10)
        stats['g_b_ratio'] = stats['media_G'] / (stats['media_B'] + 1e-10)
        
        # Entropia (medida de complexidade da textura)
        for nome, banda in zip(['R', 'G', 'B'], [red, green, blue]):
            # Calcular histograma
            banda_flat = banda[~np.isnan(banda)]
            if len(banda_flat) > 0:
                hist, _ = np.histogram(banda_flat, bins=32, density=True)
                hist = hist[hist > 0]  # Remover zeros
                if len(hist) > 0:
                    stats[f'entropy_{nome}'] = -np.sum(hist * np.log2(hist + 1e-10))
                else:
                    stats[f'entropy_{nome}'] = np.nan
            else:
                stats[f'entropy_{nome}'] = np.nan
        
        # Verificar que todos os valores são escalares
        for key, value in stats.items():
            if isinstance(value, np.ndarray):
                stats[key] = float(np.nanmean(value))
            elif not np.isscalar(value):
                stats[key] = float(value) if value is not None else np.nan
        
        return stats
    
    except Exception as e:
        print(f"⚠️ Erro ao processar polígono: {str(e)}")
        return None

print("📊 A extrair valores RGB do raster para polígonos...")

# Extrair características para todos os polígonos
features_list = []
valores_confirmado = []
poligonos_validos = []
areas = []
perimetros = []

for idx, row in gdf.iterrows():
    geom = row.geometry
    
    # Calcular características geométricas
    area = geom.area
    perimetro = geom.length
    
    # Extrair estatísticas RGB
    stats = extrair_estatisticas_poligono_rgb(geom, raster, buffer=buffer_metros)
    
    if stats is None:
        print(f"⚠️ Geometria inválida ou sem dados no índice {idx}, ignorando...")
        features_list.append({})
        valores_confirmado.append(row.get('Conf_novo', np.nan))
        poligonos_validos.append(False)
        areas.append(area)
        perimetros.append(perimetro)
        continue
    
    # Adicionar características geométricas
    stats['area'] = area
    stats['perimetro'] = perimetro
    stats['compacidade'] = 4 * np.pi * area / (perimetro ** 2) if perimetro > 0 else 0
    stats['perimetro_area_ratio'] = perimetro / area if area > 0 else 0
    
    features_list.append(stats)
    valores_confirmado.append(row.get('Conf_novo', np.nan))
    poligonos_validos.append(True)
    areas.append(area)
    perimetros.append(perimetro)
    
    if (idx + 1) % 100 == 0:
        print(f"Processados {idx + 1}/{len(gdf)} polígonos...")

# Converter para DataFrame
df_features = pd.DataFrame(features_list)
df_features['Conf_novo'] = valores_confirmado

# Remover linhas completamente vazias (polígonos inválidos)
df_features = df_features[df_features.columns[df_features.notna().any()]]

# Adicionar flag de polígonos válidos ao geodataframe
gdf['poligono_valido'] = poligonos_validos

# Guardar cópia com todas as linhas (para aplicar o modelo a todos os polígonos mais tarde)
df_todas_linhas = df_features.copy()

# Remover linhas com NaN ou Conf_novo nulo
df_features = df_features.dropna(subset=['Conf_novo'])

# Verificar % de valores NaN por coluna
nan_percent = df_features.isna().mean() * 100
print("\n=== Percentagem de valores NaN por coluna ===")
print(nan_percent[nan_percent > 0])

# Remover colunas com muitos NaNs (>50%)
colunas_para_remover = nan_percent[nan_percent > 50].index.tolist()
if colunas_para_remover:
    print(f"⚠️ A remover colunas com >50% NaNs: {', '.join(colunas_para_remover)}")
    df_features = df_features.drop(columns=colunas_para_remover)

# Armazenar cópia com as mesmas colunas de treino, para aplicar o modelo a todos os polígonos
df_inicial = df_todas_linhas.drop(columns=colunas_para_remover, errors='ignore')

print(f"\n✅ Número de polígonos com dados para treino: {len(df_features)}")
print(f"Distribuição da variável alvo: {df_features['Conf_novo'].value_counts().to_dict()}")

# === ANÁLISE EXPLORATÓRIA ===
print("\n📊 A realizar análise exploratória...")

# Verificar tipos de dados antes da correlação
print("\nVerificando tipos de dados das colunas...")
for col in df_features.columns:
    if col != 'Conf_novo':
        sample_value = df_features[col].iloc[0] if len(df_features) > 0 else None
        if sample_value is not None and not np.isscalar(sample_value):
            print(f"⚠️ Coluna '{col}' contém valores não-escalares: {type(sample_value)}")
            # Converter para escalar se necessário
            df_features[col] = df_features[col].apply(lambda x: float(np.nanmean(x)) if isinstance(x, np.ndarray) else x)

# Gravar matriz de correlação
plt.figure(figsize=(14, 12))
corr_matrix = df_features.drop(columns=['Conf_novo']).corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Matriz de Correlação das Características RGB')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'correlacao_rgb.png'))
plt.close()

# Preparar dados para modelagem
X_inicial = df_features.drop(columns='Conf_novo')
y = df_features['Conf_novo']

# Verificar e corrigir desequilíbrio de classes
class_counts = y.value_counts()
print(f"\nDistribuição de classes: {class_counts.to_dict()}")
menor_classe = class_counts.min()
maior_classe = class_counts.max()
ratio = menor_classe / maior_classe

if ratio < 0.7:
    print(f"⚠️ Desequilíbrio de classes detectado (ratio: {ratio:.2f}). Aplicando SMOTE...")
    usar_smote = True
else:
    print("✅ Distribuição de classes relativamente equilibrada.")
    usar_smote = False

# === ENGENHARIA DE CARACTERÍSTICAS E PRÉ-PROCESSAMENTO ===
print("\n🔧 A preparar características...")

# Preparar pipeline de pré-processamento
preprocessor = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler())
])

# Aplicar pré-processamento
X_prep = preprocessor.fit_transform(X_inicial)
X_prep_df = pd.DataFrame(X_prep, columns=X_inicial.columns)

# Selecionar características importantes usando RandomForest
base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
base_clf.fit(X_prep, y)

# Visualizar importância das características
feature_importance = pd.DataFrame({
    'feature': X_inicial.columns,
    'importance': base_clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Top 15 Características Mais Importantes ===")
print(feature_importance.head(15))

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Importância das Características RGB')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'importancia_caracteristicas_rgb.png'))
plt.close()

# Selecionar as melhores características
selector = SelectFromModel(base_clf, threshold='mean')
selector.fit(X_prep, y)
X_selected = selector.transform(X_prep)
selected_feat_indices = selector.get_support(indices=True)
selected_features = [X_inicial.columns[i] for i in selected_feat_indices]

print(f"\n✅ Características selecionadas ({len(selected_features)}/{len(X_inicial.columns)}):")
print(', '.join(selected_features))

# Dividir dados em treino e teste
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_selected, y, stratify=y, test_size=test_size, random_state=42
)

# Aplicar SMOTE se necessário
if usar_smote:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train_full, y_train_full)
    print(f"✅ Após SMOTE - Distribuição de classes: {pd.Series(y_train).value_counts().to_dict()}")
else:
    X_train, y_train = X_train_full, y_train_full

# === TREINO DO MODELO ===
print("\n🧠 A treinar e otimizar modelos...")

# Criar vários modelos
models = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            'n_estimators': [100, 200],
            'max_depth': [None, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', None]
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.2],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5]
        }
    }
}

# Usar validação cruzada estratificada
cv = StratifiedKFold(n_splits=num_cv_folds, shuffle=True, random_state=42)

# Treinar e avaliar modelos
best_models = {}
best_score = 0
best_model_name = ""

for name, config in models.items():
    print(f"\nOtimizando {name}...")
    
    grid = GridSearchCV(
        config["model"], 
        config["params"],
        cv=cv, 
        scoring='accuracy', 
        verbose=0,
        n_jobs=-1
    )
    
    grid.fit(X_train, y_train)
    
    print(f"Melhores parâmetros para {name}: {grid.best_params_}")
    print(f"Melhor pontuação de validação cruzada: {grid.best_score_:.4f}")
    
    # Avaliar no conjunto de teste
    y_pred = grid.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Precisão no conjunto de teste: {test_acc:.4f}")
    
    best_models[name] = grid.best_estimator_
    
    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model_name = name

# Selecionar o melhor modelo
best_model = best_models[best_model_name]
print(f"\n✅ Modelo selecionado: {best_model_name}")

# === AVALIAÇÃO DO MODELO ===
print("\n=== AVALIAÇÃO DO MODELO ===")
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

# Métricas de desempenho
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Precisão: {accuracy:.4f}")
print("\nMatriz de Confusão:")
print(conf_matrix)

# Relatório de classificação
class_report = classification_report(y_test, y_pred)
print("\nRelatório de Classificação:")
print(class_report)

# AUC-ROC (para classificação binária)
if len(np.unique(y)) == 2:
    try:
        auc_roc = roc_auc_score(y_test, y_proba[:, 1])
        print(f"\nAUC-ROC: {auc_roc:.4f}")
    except Exception as e:
        print(f"Não foi possível calcular AUC-ROC: {str(e)}")

# Validação cruzada completa
cv_scores = cross_val_score(best_model, X_selected, y, cv=cv)
print(f"\nAcurácia média (validação cruzada): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Visualizar matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Modelo RGB')
plt.savefig(os.path.join(results_dir, 'matriz_confusao_rgb.png'))
plt.close()

# === APLICAR MODELO A TODOS OS POLÍGONOS ===
print("\n📦 A classificar todos os polígonos...")

# Preparar dados originais para predição
X_todos_inicial = df_inicial.drop(columns='Conf_novo')

# Aplicar o mesmo pré-processamento
X_todos_prep = preprocessor.transform(X_todos_inicial)

# Selecionar as mesmas características
X_todos_selected = selector.transform(X_todos_prep)

# Fazer predições
todos_pred = np.full(len(gdf), np.nan)
todos_prob = np.full(len(gdf), np.nan)

# Identificar índices de polígonos válidos (com dados RGB extraídos com sucesso)
indices_validos = np.where(np.array(poligonos_validos))[0]
print(f"Polígonos com dados válidos para predição: {len(indices_validos)}/{len(gdf)}")

# Fazer predição apenas para polígonos válidos
if indices_validos.size > 0:
    X_pred_subset = X_todos_selected[indices_validos]
    preds = best_model.predict(X_pred_subset)
    probs = best_model.predict_proba(X_pred_subset)
    
    todos_pred[indices_validos] = preds
    
    if probs.shape[1] >= 2:
        todos_prob[indices_validos] = probs[:, 1]

# Adicionar resultados ao GeoDataFrame
gdf['Conf_pred'] = todos_pred
gdf['Prob_pos'] = todos_prob

# Avaliar concordância
mask_real = ~np.isnan(valores_confirmado)
mask_pred = ~np.isnan(todos_pred)
mask_ambos = mask_real & mask_pred

if np.sum(mask_ambos) > 0:
    concordancia = np.sum(np.array(valores_confirmado)[mask_ambos] == todos_pred[mask_ambos]) / np.sum(mask_ambos)
    print(f"\nConcordância (predito vs. real): {concordancia:.2%}")

# Adicionar metadados
gdf['area_m2'] = gdf.geometry.area
gdf['perim_m'] = gdf.geometry.length
gdf['compact'] = 4 * np.pi * gdf.geometry.area / (gdf.geometry.length ** 2)

# Gravar resultados
print(f"\n✅ Gravando resultados...")
gdf.to_file(
    output_gdb,
    layer=output_layer,
    driver="OpenFileGDB"
)

# Gravar também como GeoJSON
geojson_path = os.path.join(results_dir, output_layer + ".geojson")
gdf.to_file(geojson_path, driver='GeoJSON')
print(f"✅ Ficheiros guardados: {output_gdb}\\{output_layer} e {geojson_path}")

# Gravar relatório
report_path = os.path.join(results_dir, "relatorio_modelo_rgb.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(f"=== RELATÓRIO DO MODELO DE CLASSIFICAÇÃO DE ESTUFAS (RGB) ===\n")
    f.write(f"Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write(f"NOTA: Este modelo usa apenas bandas RGB (sem NIR)\n\n")
    f.write(f"Modelo selecionado: {best_model_name}\n")
    f.write(f"Parâmetros: {best_model.get_params()}\n\n")
    f.write(f"Características selecionadas ({len(selected_features)}):\n")
    f.write(", ".join(selected_features) + "\n\n")
    f.write(f"Acurácia no teste: {accuracy:.4f}\n")
    f.write(f"Acurácia média (CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
    f.write("Matriz de Confusão:\n")
    f.write(str(conf_matrix) + "\n\n")
    f.write("Relatório de Classificação:\n")
    f.write(class_report + "\n")
    if len(np.unique(y)) == 2 and 'auc_roc' in locals():
        f.write(f"AUC-ROC: {auc_roc:.4f}\n\n")
    f.write(f"Total de polígonos no shapefile: {len(gdf)}\n")
    f.write(f"Polígonos com dados válidos para predição: {len(indices_validos)}\n")
    if np.sum(mask_ambos) > 0:
        f.write(f"Concordância (predito vs. real): {concordancia:.2%}\n")

print(f"\n✅ Relatório guardado em: {report_path}")
print(f"\n✅ Análise completa! Resultados guardados em: {results_dir}")