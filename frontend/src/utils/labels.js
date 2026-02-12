export const RISK_LEVELS = ["Low", "Medium", "High"];

export const VALUE_LABELS = {
  gender: { f: "Female", m: "Male" },
  managerGender: { f: "Female", m: "Male" },
  coachingSupport: { yes: "Has coach", no: "No coach", "my head": "Direct manager" },
  compensationType: { white: "Formal pay", grey: "Informal pay" },
  commuteMethod: { bus: "Bus", car: "Car", foot: "Walk" },
  recruitmentSource: {
    youjs: "Online job board",
    empjs: "Employer career site",
    rabrecNErab: "Recruiter/network",
    recNErab: "Recruiter channel",
    referal: "Employee referral",
    friends: "Friend referral",
    advert: "Advertisement",
    KA: "Campus/agency",
  },
};

export function prettyValue(field, value) {
  if (value === null || value === undefined || value === "") return "Unknown";
  return VALUE_LABELS[field]?.[String(value)] ?? String(value);
}
