/// Return the fields in one comma-separated row.
///
/// This deliberately small fixture stands in for a repository an operator is
/// investigating. It is not part of the Piku workspace.
pub fn fields(row: &str) -> Vec<&str> {
    row.split(',').filter(|field| !field.is_empty()).collect()
}

