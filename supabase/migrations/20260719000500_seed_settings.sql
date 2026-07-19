-- Sainik Sahayak · migration 5: default app_settings (admin-editable at runtime)

insert into public.app_settings (key, value) values
  ('not_found_message', jsonb_build_object(
    'en', 'This information is not available in the knowledge base. Please contact your unit''s concerned office.',
    'hi', 'यह जानकारी उपलब्ध ज्ञानकोश में नहीं है। कृपया अपनी यूनिट के संबंधित कार्यालय से संपर्क करें।'
  )),
  ('escalation_contact', jsonb_build_object(
    'en', '', 'hi', ''
  )),
  ('rerank_refusal_threshold', to_jsonb(0.35)),
  ('message_retention_days',   to_jsonb(90)),
  ('audit_retention_days',     to_jsonb(365)),
  ('sample_questions', jsonb_build_array(
    jsonb_build_object('en', 'How do I apply for annual leave?',
                       'hi', 'वार्षिक अवकाश के लिए आवेदन कैसे करूँ?'),
    jsonb_build_object('en', 'What documents are needed for an ECHS card?',
                       'hi', 'ECHS कार्ड के लिए कौन से दस्तावेज़ चाहिए?'),
    jsonb_build_object('en', 'How is family pension claimed?',
                       'hi', 'पारिवारिक पेंशन का दावा कैसे किया जाता है?'),
    jsonb_build_object('en', 'What are the AGIF loan eligibility rules?',
                       'hi', 'AGIF ऋण पात्रता के नियम क्या हैं?')
  ))
on conflict (key) do nothing;
