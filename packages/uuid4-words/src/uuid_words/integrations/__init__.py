"""Optional integration helpers for specific SDKs / frameworks.

Nothing in ``uuid_words`` core imports from this subpackage, and nothing
here is imported automatically -- import the module you need explicitly
(e.g. ``from uuid_words.integrations.openai import WordsUUID``) so that
optional dependencies (like pydantic) are only required when you actually
use them.
"""
