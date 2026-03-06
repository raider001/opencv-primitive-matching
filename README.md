# Pattern Matching Algorithm using OpenPNP (OpenCV)

This project implements a basic pattern matching algorithm using OpenCV via the OpenPNP library.

## Overview

The pattern matching algorithm finds occurrences of a template image within a larger source image. It's useful for computer vision applications such as:
- Visual quality inspection
- Object detection
- Component localization in manufacturing

## Core Components

### 1. **IO.java**
A utility class providing console output methods.

### 2. **PatternMatcher.java**
The main pattern matching library with the following key methods:

#### Basic Template Matching
```java
// Match a template in source image (file paths)
MatchResult result = PatternMatcher.matchTemplate("source.jpg", "template.jpg");

// Match a template in source image (Mat objects)
Mat source = Imgcodecs.imread("source.jpg");
Mat template = Imgcodecs.imread("template.jpg");
MatchResult result = PatternMatcher.matchTemplate(source, template);
```

**Returns:** `MatchResult` containing:
- `confidence`: Match confidence (0.0 to 1.0, where 1.0 = perfect match)
- `location`: Point(x, y) coordinates of the match in the source image
- `templateSize`: Size of the template

#### Find All Matches Above Threshold
```java
// Find all matches above 80% confidence
List<MatchResult> matches = PatternMatcher.findAllMatches(
    source, template, 0.80);
```

**Returns:** List of `MatchResult` objects sorted by confidence (descending)

#### Multi-Scale Matching
```java
// Match at multiple scales/sizes
double[] scales = {0.5, 0.75, 1.0, 1.25, 1.5};
MatchResult result = PatternMatcher.matchTemplateMultiScale(
    source, template, scales);
```

**Returns:** `MatchResult` with the best match across all scales

### 3. **MatchResult.java** (Inner Class)
Represents a single template match with:
- Confidence score (0.0-1.0)
- Location coordinates (x, y)
- Template size (width, height)

## Usage Example

```java
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;

public class PatternMatchingExample {
    public static void main(String[] args) {
        // Load images
        Mat source = Imgcodecs.imread("image.jpg");
        Mat template = Imgcodecs.imread("pattern.jpg");
        
        // Find the best match
        PatternMatcher.MatchResult result = PatternMatcher.matchTemplate(source, template);
        
        System.out.println(result);
        // Output: Match{confidence=92.50%, location=(150.0,200.0), size=50.0x50.0}
        
        // Find all matches above 80% confidence
        List<PatternMatcher.MatchResult> allMatches = 
            PatternMatcher.findAllMatches(source, template, 0.80);
        
        for (PatternMatcher.MatchResult match : allMatches) {
            System.out.println(match);
        }
    }
}
```

## How It Works

1. **Template Matching Algorithm**: Uses normalized cross-correlation coefficient (TM_CCOEFF_NORMED) to measure similarity between the template and each region of the source image.

2. **Confidence Score**: The algorithm returns a confidence value between -1 and 1, where:
   - 1.0 = Perfect match
   - 0.0 = No correlation
   - -1.0 = Perfect inverse match

3. **Multi-Scale Matching**: Tests the template at different scales, useful when the pattern size is unknown.

## Dependencies

- **OpenCV 4.7.0** via OpenPNP library
- **Java 25+**

### Maven Dependencies

```xml
<dependency>
    <groupId>org.openpnp</groupId>
    <artifactId>opencv</artifactId>
    <version>4.7.0-0</version>
</dependency>
<dependency>
    <groupId>nu.pattern</groupId>
    <artifactId>opencv</artifactId>
    <version>4.7.0-0</version>
</dependency>
```

## Building the Project

```bash
mvn clean compile
```

## Running the Demo

```bash
mvn exec:java -Dexec.mainClass="org.example.Main"
```

## Performance Considerations

1. **Image Size**: Larger images require more computation. Consider resizing if necessary.
2. **Template Size**: Smaller templates are faster to match but less specific.
3. **Confidence Threshold**: Higher thresholds are faster (fewer matches to process).
4. **Scale Factors**: Fewer scales = faster processing but may miss matches.

## Limitations

- Works best with similar lighting and scale
- Requires exact pixel-level features for good matches
- Not rotation-invariant (template and source must have same orientation)
- Memory usage grows with image size

## Future Enhancements

- Support for rotation-invariant matching
- GPU acceleration using CUDA
- Multi-threaded matching for large images
- Feature-based matching (SIFT, SURF)
- Perspective transformation handling

