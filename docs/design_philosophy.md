# DocFrame vs GeoPandas: Design Philosophy Comparison

## Overview

DocFrame is inspired by GeoPandas' design philosophy of extending a high-performance data library (Polars instead of pandas) with domain-specific functionality (text analysis instead of geospatial operations). This document explores the design parallels and explains why DocFrame follows this proven pattern.

## Design Philosophy Comparison

### GeoPandas Approach
- **Base Library**: Extends pandas DataFrame/Series with geometry support
- **Special Column**: `geometry` column containing geospatial objects (points, polygons, lines)
- **Domain Methods**: Spatial operations (buffer, intersection, union, dissolve, etc.)
- **Performance**: Leverages pandas + specialized geospatial libraries (GEOS, GDAL, PROJ)
- **Namespace Integration**: `.sindex` for spatial indexing, spatial join operations

### DocFrame Approach
- **Base Library**: Extends Polars DataFrame/LazyFrame with document support
- **Special Column**: `document` column containing text documents (auto-detected)
- **Domain Methods**: Text operations (tokenize, clean, word_count, ngrams, etc.)
- **Performance**: Leverages Polars + specialized text libraries (spaCy, NLTK optional)
- **Namespace Integration**: `.text` namespace for unified text processing across all Polars types

## Class Structure Comparison

### GeoPandas Pattern
```python
import geopandas as gpd
import pandas as pd

# GeoSeries - geometry-aware Series
geo_series = gpd.GeoSeries([point1, point2, polygon1])
areas = geo_series.area  # Geometric property
buffered = geo_series.buffer(100)  # Spatial operation

# GeoDataFrame - DataFrame with special geometry column
gdf = gpd.GeoDataFrame(data, geometry='geom_column')
gdf.plot()  # Specialized visualization
nearby = gdf[gdf.intersects(target_geom)]  # Spatial filtering

# Preserves DataFrame operations while adding geospatial capabilities
merged = gdf.merge(other_df, on='id')  # Still a GeoDataFrame
```

### DocFrame Pattern
```python
import docframe as dp
import polars as pl

# DocDataFrame - DataFrame with special document column
df = dp.DocDataFrame(data)  # Auto-detects document column
print(df.active_document_name)  # Shows detected column

# DocLazyFrame - LazyFrame equivalent for memory efficiency
lazy_df = dp.DocLazyFrame(lazy_data, document_column="content")

# Preserves DataFrame operations while adding text capabilities
merged = df.join(other_df, on='id')  # Still a DocDataFrame
filtered = df.filter(pl.col('category') == 'news')  # Polars operations work

# Text-specific operations
stats = df.add_word_count().describe_text()
processed = df.clean_documents().filter_by_length(min_words=10)
```

## Key Design Principles

### 1. Composition Over Inheritance

**GeoPandas**: Wraps pandas objects while maintaining their interface
```python
class GeoDataFrame:
    def __init__(self, data, geometry=None):
        # Composition: contains a pandas DataFrame
        self._df = pd.DataFrame(data)
        self._geometry = geometry
```

**DocFrame**: Wraps Polars objects while maintaining their interface
```python
class DocDataFrame:
    def __init__(self, data, document_column=None):
        # Composition: contains a Polars DataFrame
        self._df = pl.DataFrame(data)
        self._document_column = document_column or self.guess_document_column(self._df)
```

### 2. Transparent Operations

**GeoPandas**: All pandas operations work seamlessly
```python
gdf['new_column'] = gdf['existing'] * 2  # Pandas operation
gdf.groupby('category').sum()            # Pandas groupby
gdf.buffer(100)                          # GeoPandas operation
```

**DocFrame**: All Polars operations work seamlessly
```python
df = df.with_columns(pl.col('price') * 1.1)  # Polars operation
df.group_by('category').agg(pl.sum('count'))  # Polars groupby  
df.add_word_count()                           # DocFrame operation
```

### 3. Special Column Pattern

**GeoPandas**: Geometry column is central to spatial operations
```python
gdf.geometry.area          # Access geometry properties
gdf.set_geometry('new_col') # Switch geometry column
gdf.geometry.plot()        # Visualize geometries
```

**DocFrame**: Document column is central to text operations
```python
df.document.text.word_count()  # Access text properties
df.set_document('new_col')     # Switch document column
df.describe_text()             # Analyze text statistics
```

### 4. Auto-Detection Intelligence

**GeoPandas**: Automatically handles geometry column detection
```python
# GeoPandas finds geometry columns automatically
gdf = gpd.read_file('shapefile.shp')  # Auto-detects geometry
```

**DocFrame**: Automatically detects document columns using text length heuristic
```python
# DocFrame finds document columns using longest average text length
df = dp.read_csv('documents.csv')  # Auto-detects best text column
best_col = dp.DocDataFrame.guess_document_column(polars_df)
```

## Namespace Integration Philosophy

### GeoPandas Spatial Index
```python
# Spatial index for performance
gdf.sindex.intersection(bounds)
gdf.sindex.nearest(geometry)
```

### DocFrame Text Namespace
```python
import polars as pl
import docframe  # Registers text namespace

# Text namespace works across all Polars types
df.select(pl.col('text').text.tokenize())     # Expression
series.text.word_count()                       # Series
df.text.add_word_count('content')             # DataFrame
lazy_df.text.clean('text').collect()          # LazyFrame
```

## Performance Philosophy

### GeoPandas Performance Strategy
- **Vectorized operations** through pandas
- **Specialized libraries** (GEOS for geometry, GDAL for I/O)
- **Spatial indexing** for efficient spatial queries
- **CRS handling** for coordinate system transformations

### DocFrame Performance Strategy
- **Vectorized operations** through Polars
- **Lazy evaluation** for memory efficiency
- **Columnar storage** for text data
- **Query optimization** through Polars' optimizer
- **Parallel processing** for text operations

## Why This Design Works

### Proven Pattern Success
1. **GeoPandas** has become the standard for geospatial analysis in Python
2. **Familiar API** reduces learning curve for users
3. **Ecosystem compatibility** with existing Polars tools and workflows
4. **Composable design** allows mixing domain operations with general data operations

### DocFrame Advantages
1. **Memory efficiency** through Polars lazy evaluation
2. **Performance** through Polars' columnar architecture  
3. **Type safety** through Polars' type system
4. **Modern API** with expression-based operations
5. **Unified namespace** across all data types

## Text vs Spatial Parallels

| Aspect | GeoPandas (Spatial) | DocFrame (Text) |
|--------|-------------------|------------------|
| **Domain Objects** | Points, Polygons, Lines | Documents, Tokens, N-grams |
| **Core Operations** | Buffer, Intersection, Union | Tokenize, Clean, Count |
| **Analysis Methods** | Spatial Join, Dissolve | Concordance, DTM, Statistics |
| **Filtering** | Spatial predicates | Pattern matching, length filtering |
| **Visualization** | Maps, spatial plots | Word clouds, text statistics |
| **Performance** | Spatial indexes | Text indexes, lazy evaluation |

## Composition Benefits

### Flexibility
```python
# DocFrame: Easy to add new text methods without changing core
class DocDataFrame:
    def add_custom_text_analysis(self):
        # Add domain-specific method
        return self.with_columns(custom_analysis(self.document))

# Users can extend functionality easily
df.add_custom_text_analysis().filter_by_pattern("research")
```

### Maintainability
```python
# Changes to Polars automatically benefit DocFrame
# No need to reimplement core DataFrame functionality
# Focus development on text-specific features
```

### Testing
```python
# Can test text functionality separately from DataFrame operations
# Polars handles data operations, DocFrame handles text operations
# Clear separation of concerns
```

## Evolution and Future

### GeoPandas Evolution
- Started with basic spatial operations
- Added advanced analysis (spatial joins, CRS handling)
- Integrated with visualization ecosystem (matplotlib, folium)
- Performance improvements through better indexing

### DocFrame Roadmap
- **Current**: Basic text operations, document column management
- **Near-term**: Advanced NLP integration, performance optimization
- **Future**: Machine learning integration, distributed processing
- **Ecosystem**: Integration with text visualization, ML libraries

## Conclusion

DocFrame successfully adapts GeoPandas' proven design philosophy to text analysis:

1. **Familiar patterns** make adoption easy for data scientists
2. **Modern backend** (Polars) provides performance advantages
3. **Extensible design** allows growth and customization
4. **Clear separation** between data operations and domain operations
5. **Unified API** across eager and lazy evaluation

This design philosophy ensures DocFrame feels familiar while providing cutting-edge performance for text analysis workflows.
```

## Key Similarities

### 1. **Specialized Accessors**
- **GeoPandas**: `.geometry` accessor for spatial operations
- **ATAPCorpus**: `.document` accessor for text operations

### 2. **Domain-Specific Methods**
- **GeoPandas**: `buffer()`, `intersects()`, `within()`, `area`, etc.
- **ATAPCorpus**: `tokenize()`, `clean()`, `word_count()`, `contains_pattern()`, etc.

### 3. **Seamless Integration**
- **GeoPandas**: Works with pandas operations + spatial operations
- **ATAPCorpus**: Works with polars operations + text operations

### 4. **Specialized Constructors**
- **GeoPandas**: `from_file()`, `from_postgis()`, etc.
- **ATAPCorpus**: `from_texts()`, `from_csv()`, etc.

### 5. **Performance Optimization**
- **GeoPandas**: Vectorized spatial operations
- **ATAPCorpus**: Vectorized text operations using polars

## Key Differences

### Backend Choice
- **GeoPandas**: Built on pandas (row-oriented, mature ecosystem)
- **ATAPCorpus**: Built on polars (column-oriented, better performance)

### Domain Focus
- **GeoPandas**: Geospatial analysis (2D/3D coordinates, projections, topology)
- **ATAPCorpus**: Text analysis (tokenization, NLP, pattern matching)

### Visualization
- **GeoPandas**: Rich mapping capabilities with matplotlib/folium
- **ATAPCorpus**: Focus on text analysis (could integrate with text visualization tools)

## Code Pattern Comparison

### Creating from Data
```python
# GeoPandas
from shapely.geometry import Point
points = [Point(1, 2), Point(3, 4)]
gdf = gpd.GeoDataFrame({'id': [1, 2]}, geometry=points)

# ATAPCorpus  
texts = ["Hello world", "Text analysis"]
df = atp.DocDataFrame.from_texts(texts, {'id': [1, 2]})
```

### Domain-Specific Operations
```python
# GeoPandas - Spatial operations
gdf.geometry.area                    # Calculate areas
gdf.geometry.buffer(100)             # Create buffers
gdf[gdf.geometry.intersects(polygon)] # Spatial filter

# ATAPCorpus - Text operations
df.document.word_count()             # Count words
df.document.clean()                  # Clean text
df.filter_by_pattern("hello")       # Text filter
```

### Aggregation
```python
# GeoPandas
gdf.dissolve(by='category')          # Spatial aggregation

# ATAPCorpus
df.groupby_agg('category', text_agg='concat')  # Text aggregation
```

## Performance Advantages

### GeoPandas
- Mature ecosystem with optimized spatial libraries
- Extensive visualization capabilities
- Well-integrated with scientific Python stack

### ATAPCorpus
- **Polars backend**: Faster operations, better memory usage
- **Lazy evaluation**: Process large datasets efficiently
- **Column-oriented**: Better for analytical workloads
- **Modern API**: More consistent and intuitive

## Use Case Scenarios

### GeoPandas Ideal For:
- Spatial analysis and GIS workflows
- Map visualization and cartography
- Geometric computations and topology analysis
- Integration with spatial databases

### ATAPCorpus Ideal For:
- Large-scale text processing and analysis
- NLP pipeline preprocessing
- Text mining and document analysis
- Performance-critical text operations

## Example: Equivalent Workflows

### GeoPandas Workflow
```python
import geopandas as gpd

# Load spatial data
gdf = gpd.read_file("data.shp")

# Add calculated columns
gdf['area'] = gdf.geometry.area
gdf['perimeter'] = gdf.geometry.length

# Filter by spatial criteria
large_areas = gdf[gdf.area > 1000]

# Spatial aggregation
dissolved = gdf.dissolve(by='category')

# Export results
dissolved.to_file("results.shp")
```

### ATAPCorpus Workflow
```python
import atapcorpus as atp

# Load text data
df = atp.DocDataFrame.from_csv("data.csv", document_column="text")

# Add calculated columns
df = df.add_word_count().add_char_count()

# Filter by text criteria  
long_docs = df.filter_by_length(min_words=100)

# Text aggregation
grouped = long_docs.groupby_agg('category', text_agg='concat')

# Export results
grouped.write_csv("results.csv")
```

## Why Composition Instead of Inheritance?

You might wonder why ATAPCorpus uses composition (wrapping) instead of direct inheritance like GeoPandas. Here's the key difference:

### GeoPandas Approach (Direct Inheritance)
```python
class GeoSeries(pd.Series):
    """Directly inherits from pandas Series"""
    def buffer(self, distance):
        # All pandas methods work automatically
        return self.apply(buffer_function)

class GeoDataFrame(pd.DataFrame):
    """Directly inherits from pandas DataFrame"""
    def __init__(self, data, geometry=None):
        super().__init__(data)
        self._geometry_column_name = geometry  # Can add attributes
```

### ATAPCorpus Approach (Composition)
```python
class DocSeries:
    """Wraps polars Series"""
    def __init__(self, data):
        self._series = pl.Series(data)  # Composition, not inheritance
    
    def tokenize(self):
        return self._series.map_elements(tokenize_func)
```

### Technical Reasons for This Choice

#### 1. **Polars Architecture Limitations**
- **Immutable objects**: Polars objects can't have custom attributes added
- **No subclassing support**: Polars wasn't designed for inheritance
- **Rust backend**: Core objects are Rust structs, not Python classes

#### 2. **Pandas vs Polars Design Philosophy**
```python
# pandas - mutable, inheritance-friendly
series = pd.Series([1, 2, 3])
series.custom_attr = "value"  # ✅ Works fine

# polars - immutable, composition-oriented  
series = pl.Series([1, 2, 3])
series.custom_attr = "value"  # ❌ AttributeError
```

#### 3. **Method Return Type Challenges**
With inheritance, every polars method would need to return our custom type:
```python
class DocDataFrame(pl.DataFrame):  # Hypothetical
    def filter(self, condition):
        result = super().filter(condition)  # Returns pl.DataFrame
        # How do we convert back to DocDataFrame automatically?
        return DocDataFrame(result)  # Would need this everywhere
```

#### 4. **API Stability**
Polars is evolving rapidly. Composition provides a stable interface that doesn't break when polars internals change.

### Trade-offs

#### Advantages of Composition (Our Approach)
- ✅ **Stable**: Insulated from polars API changes
- ✅ **Clean namespace**: No method name conflicts
- ✅ **Controlled interface**: We decide what to expose
- ✅ **Type safety**: Clear separation of concerns

#### Advantages of Inheritance (GeoPandas Approach)
- ✅ **Drop-in replacement**: `isinstance(obj, pl.DataFrame)` works
- ✅ **All methods available**: Every polars method works automatically
- ✅ **Familiar**: Users expect inheritance pattern
- ✅ **Less code**: No need to delegate methods

### Alternative: Hybrid Approach

If polars supported subclassing better, we could use a hybrid approach:
```python
class DocDataFrame(pl.DataFrame):
    def __init_subclass__(cls):
        # Register text methods automatically
        super().__init_subclass__()
    
    @property 
    def document(self):
        return DocSeries(self[self._document_column])
    
    # All polars methods work automatically
    # Our text methods are additions
```

But this isn't possible with current polars architecture.

## Corrected: Document Column as Shortcut Property

**Important clarification**: Just like in GeoPandas, the `document` property is a **shortcut accessor** to whichever column is designated as the document column, not a hardcoded column name.

### GeoPandas Pattern
```python
import geopandas as gpd

# Column can be named anything
gdf = gpd.GeoDataFrame(data, geometry='my_geom_column')  # or 'shape', 'boundaries', etc.

# .geometry always accesses the designated geometry column
gdf.geometry  # -> accesses 'my_geom_column'
```

### ATAPCorpus Pattern (Corrected)
```python
import atapcorpus as atp

# Column can be named anything
df = atp.DocDataFrame(data, document_column='text')  # or 'content', 'article', etc.

# .document always accesses the designated document column
df.document  # -> accesses 'text' column
df.document_column  # -> returns 'text'
```

### Examples
```python
# Using 'article_text' as document column
df = atp.DocDataFrame.from_csv('news.csv', document_column='article_text')
df.document  # accesses the 'article_text' column
df.document.word_count()  # operates on 'article_text'

# Using 'content' as document column  
df2 = atp.DocDataFrame(data, document_column='content')
df2.document  # accesses the 'content' column

# Operations preserve the document column reference
df3 = df2.add_word_count()
df3.document_column  # still returns 'content'
```

This matches GeoPandas exactly - the special property (`.geometry` or `.document`) is a shortcut to access whichever column has been designated as the special column, regardless of its actual name.

## Automatic Document Column Detection

One of the key usability features of ATAPCorpus is automatic document column detection. When creating a `DocDataFrame` without specifying the `document_column` parameter, the library intelligently identifies the most likely document column.

### Detection Algorithm

```python
@classmethod
def guess_document_column(cls, df: pl.DataFrame, sample_size: int = 1000) -> Optional[str]:
    """Guess the best document column by finding the string column with the longest average length."""
```

The algorithm works as follows:

1. **Filter to string columns**: Only `pl.Utf8` columns are considered as potential document columns
2. **Sample data for performance**: Uses the first `sample_size` rows (default 1000) to avoid processing huge datasets
3. **Calculate average character length**: For each string column, computes the mean character length using polars' `.str.len_chars().mean()`
4. **Select the longest**: Returns the column name with the highest average character length
5. **Handle edge cases**: Returns `None` if no string columns exist

### Integration with Constructor

```python
def __init__(self, data: Union[pl.DataFrame, Dict[str, Any], None] = None,
             document_column: Optional[str] = None):
    # ... data processing ...
    
    # Auto-detect document column if not specified
    if document_column is None:
        guessed_column = self.guess_document_column(self._df)
        if guessed_column is not None:
            document_column = guessed_column
        else:
            document_column = self._DEFAULT_DOCUMENT_COLUMN  # 'document'
```

This approach:
- **Maintains backward compatibility**: Existing code with explicit `document_column` works unchanged
- **Improves user experience**: Reduces boilerplate in common cases
- **Provides transparency**: Users can call `guess_document_column()` directly to see what would be detected
- **Handles edge cases gracefully**: Falls back to sensible defaults when auto-detection isn't possible

### Design Rationale

This feature was inspired by pandas' automatic index detection and similar conveniences in data science libraries. The goal is to minimize friction for users working with obvious document datasets while maintaining full control when needed.

The choice to use average character length as the heuristic is based on the observation that document columns typically contain substantially longer text than metadata columns (titles, categories, IDs, etc.).

## Conclusion

The composition approach was chosen due to polars' technical constraints, not design preference. If polars supported inheritance like pandas, we would absolutely use the GeoPandas approach for better API consistency.

ATAPCorpus successfully adapts GeoPandas' proven design philosophy to the text analysis domain:

1. **Extends high-performance library** (polars vs pandas)
2. **Adds domain-specific column** (document vs geometry)
3. **Provides specialized methods** (text processing vs spatial operations)
4. **Maintains underlying functionality** (polars operations vs pandas operations)
5. **Offers domain-specific constructors** and export methods

This approach provides:
- **Familiar API** for users of GeoPandas/pandas
- **Performance benefits** from polars backend
- **Specialized functionality** for text analysis
- **Seamless integration** with existing polars workflows
