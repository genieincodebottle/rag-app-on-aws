output "api_url" {
  value = module.api.invoke_url
}

output "user_pool_id" {
  value = module.auth.user_pool_id
}

output "user_pool_client_id" {
  value = module.auth.client_id
}

output "documents_bucket" {
  value = module.storage.bucket_name
}

output "migrate_function_name" {
  value = module.migrate_function.function_name
}

output "function_names" {
  value = {
    api     = module.api_function.function_name
    ingest  = module.ingest_function.function_name
    migrate = module.migrate_function.function_name
  }
}

output "function_versions" {
  description = "Published versions behind the live aliases; the deploy workflow records these for rollback."
  value = {
    api     = module.api_function.version
    ingest  = module.ingest_function.version
    migrate = module.migrate_function.version
  }
}

output "dashboard_name" {
  value = module.observability.dashboard_name
}

output "ingest_dlq" {
  value = module.storage.dlq_name
}
