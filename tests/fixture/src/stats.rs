/// Compute statistics over a slice of integers.

/// Returns the sum of all elements.
pub fn sum(values: &[i32]) -> i32 {
    values.iter().sum()
}

/// Returns the minimum value, or None if the slice is empty.
pub fn min(values: &[i32]) -> Option<i32> {
    values.iter().copied().reduce(i32::min)
}

/// Returns the maximum value, or None if the slice is empty.
pub fn max(values: &[i32]) -> Option<i32> {
    values.iter().copied().reduce(i32::max)
}

/// Returns the arithmetic mean of all elements.
///
/// BUG: panics on empty slice (divide-by-zero).
/// Scenario: ask piku to find and fix the panic.
pub fn mean(values: &[i32]) -> f64 {
    let n = values.len(); // BUG: no empty-check here
    let s: i32 = values.iter().sum();
    s as f64 / n as f64 // panics (NaN / 0.0 = NaN, but integer divide panics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_empty() {
        assert_eq!(sum(&[]), 0);
    }

    #[test]
    fn sum_positive() {
        assert_eq!(sum(&[1, 2, 3, 4, 5]), 15);
    }

    // MISSING: test for mean() — no coverage of the panic case
    // Scenario: ask piku to add a test that catches the empty-slice panic.
}
