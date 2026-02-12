use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemFn, LitInt, parse_macro_input};

/// A drop-in replacement for `#[test]` that prints elapsed time and fails
/// any test exceeding a timeout (default: 1 second).
///
/// # Usage
/// ```ignore
/// use test_macros::timed_test;
///
/// #[timed_test]
/// fn my_test() {
///     assert_eq!(2 + 2, 4);
/// }
///
/// #[timed_test(300)]
/// fn slow_test() {
///     // This test gets a 300-second timeout
/// }
/// ```
#[proc_macro_attribute]
pub fn timed_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    let timeout_secs: u64 = if attr.is_empty() {
        1
    } else {
        let lit = parse_macro_input!(attr as LitInt);
        lit.base10_parse::<u64>()
            .expect("timed_test expects an integer timeout in seconds")
    };

    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name = &input_fn.sig.ident;
    let fn_block = &input_fn.block;
    let fn_attrs = &input_fn.attrs;
    let fn_vis = &input_fn.vis;

    let expanded = quote! {
        #(#fn_attrs)*
        #[test]
        #fn_vis fn #fn_name() {
            let __timed_start = ::std::time::Instant::now();

            let __timed_result = ::std::panic::catch_unwind(
                ::std::panic::AssertUnwindSafe(|| #fn_block)
            );

            let __timed_elapsed = __timed_start.elapsed();

            eprintln!(
                "[timer] {} completed in {:.3}s",
                stringify!(#fn_name),
                __timed_elapsed.as_secs_f64()
            );

            if __timed_elapsed.as_secs() >= #timeout_secs {
                panic!(
                    "[timer] {} exceeded {}s timeout ({:.3}s)",
                    stringify!(#fn_name),
                    #timeout_secs,
                    __timed_elapsed.as_secs_f64()
                );
            }

            if let ::std::result::Result::Err(__timed_payload) = __timed_result {
                ::std::panic::resume_unwind(__timed_payload);
            }
        }
    };

    expanded.into()
}
